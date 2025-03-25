"""
Основной скрипт для обучения и оценки модели классификации твитов
"""

import os
import argparse
import mlflow
from typing import Tuple
import lightgbm
import pandas as pd

from data_loader import load_data
from feature_extraction.feature_manager import FeatureExtractor
from model.tokenizator_model import TokenizatorModel
from visualization.feature_viz import visualize_feature_importance
from visualization.performance_viz import (visualize_confusion_matrix, visualize_roc_curve,
                                           visualize_pr_curve, visualize_calibration_curve,
                                           visualize_learning_curve, visualize_lift)
from visualization.shap_viz import visualize_shap_values
from utils.logging_utils import setup_logger
from config import DEFAULT_RANDOM_STATE, DEFAULT_THRESHOLD, DEFAULT_DATA_FILE

logger = setup_logger()


def train_tokenizator_model(
        data_file: str,
        output_dir: str,
        threshold: float = DEFAULT_THRESHOLD,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_splits: int = 5,
        test_size: float = 0.2) -> Tuple[TokenizatorModel, pd.DataFrame, pd.Series]:
    """
    Обучение и оценка модели классификации твитов

    Args:
        data_file: Путь к файлу с данными
        output_dir: Директория для сохранения результатов
        threshold: Порог для бинаризации
        random_state: Seed для генератора случайных чисел
        n_splits: Количество фолдов для кросс-валидации
        test_size: Размер тестовой выборки (0-1)

    Returns:
        Tuple: Обученная модель, тестовые признаки и тестовые целевые значения
    """
    # Создание директории для результатов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Запуск отслеживания MLflow
    mlflow.start_run(run_name="tokenizator_model")

    try:
        # 1. Загрузка датасета
        logger.info(f"Загрузка датасета из {data_file}...")
        df = load_data(data_file)

        # Логирование информации о датасете
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("positive_ratio", (df['total_trade_volume'] >= threshold).mean())

        # 2. Извлечение признаков
        logger.info("Извлечение признаков...")
        feature_extractor = FeatureExtractor()
        features_df, invalid_indices = feature_extractor.extract_all_features(df)

        # Удаление невалидных постов
        if invalid_indices:
            logger.info(f"Удаление {len(invalid_indices)} невалидных постов с недоступными изображениями")
            valid_df = df.drop(invalid_indices)
            valid_features_df = features_df.drop(invalid_indices)
        else:
            valid_df = df
            valid_features_df = features_df

        # 3. Инициализация модели
        logger.info("Инициализация модели...")
        model = TokenizatorModel(random_state=random_state)

        # 4. Предобработка данных
        logger.info("Предобработка данных...")
        X = valid_features_df.copy()
        y = valid_df['total_trade_volume']
        X, y_binary = model.preprocess_data(X, y, threshold=threshold)

        # 5. Удаление сильно коррелирующих признаков
        logger.info("Удаление сильно коррелирующих признаков...")
        X_reduced, dropped_features = model.analyze_feature_correlations(X, threshold=0.9)

        # 6. Группировка признаков по типу
        logger.info("Группировка признаков по типу...")
        feature_groups = model.group_features(X_reduced)

        # 7. Разделение данных с использованием кросс-валидации
        logger.info(f"Разделение данных с {n_splits}-фолдовой кросс-валидацией...")
        X_train_val, X_test, y_train_val, y_test, kfold = model.split_data(
            X_reduced, y_binary, n_splits=n_splits, test_size=test_size)

        # 8. Анализ распределения классов
        class_stats = model.analyze_class_distribution(y_train_val)
        mlflow.log_params({
            'imbalance_ratio': class_stats['imbalance_ratio'],
            'uniformity_index': class_stats['uniformity_index']
        })

        # 9. Определение оптимального метода обработки дисбаланса
        best_imbalance_method, imbalance_params = model.select_optimal_imbalance_method(
            X_train_val, y_train_val, kfold)
        mlflow.log_param('imbalance_method', best_imbalance_method)

        # 10. Обработка дисбаланса с использованием лучшего метода
        X_train_val_balanced, y_train_val_balanced, _ = model.handle_class_imbalance(
            X_train_val, y_train_val, method=best_imbalance_method)

        # 11. Совместная оптимизация гиперпараметров и отбора признаков
        logger.info("Запуск совместной оптимизации гиперпараметров и отбора признаков...")
        best_params, selected_features = model.optimize_jointly(
            X_train_val_balanced, y_train_val_balanced,
            cv_outer=3, cv_inner=5, n_trials=30
        )

        # 12. Отбор признаков с оптимальными параметрами
        logger.info("Отбор признаков с использованием оптимальных параметров...")
        X_train_val_selected, selected_features = model.select_features_with_optimal_parameters(
            X_train_val_balanced, y_train_val_balanced,
            params=best_params, trial_budget=50
        )

        # Обновляем тестовый набор с учетом выбранных признаков
        X_test_selected = X_test[selected_features]

        # 13. Обучение модели с выбранными признаками и оптимальными гиперпараметрами
        logger.info("Обучение модели с выбранными признаками и оптимальными гиперпараметрами...")
        model.train(
            X_train_val_selected, y_train_val_balanced, kfold,
            params=best_params, threshold=model.best_threshold
        )

        # 14. Анализ стабильности результатов между фолдами
        logger.info("Анализ стабильности результатов между фолдами...")
        stability_metrics = model.analyze_fold_stability()

        # 15. Переобучение финальной модели на тренировочных и валидационных данных
        logger.info("Переобучение финальной модели на тренировочных и валидационных данных...")
        model.train_final_model(
            X_train_val_selected,
            y_train_val_balanced,
            params=best_params
        )

        # 16. Оценка модели на тестовой выборке
        logger.info("Оценка модели на тестовой выборке...")
        test_metrics = model.evaluate(X_test_selected, y_test, threshold=model.best_threshold)

        # Логирование метрик
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        # 17. Расчет и визуализация лифта
        logger.info("Расчет и визуализация лифта...")
        bin_metrics = model.compute_lift(X_test_selected, y_test, bins=10)
        visualize_lift(bin_metrics, output_path=os.path.join(output_dir, 'lift_charts.png'))

        # 18. Визуализация важности признаков
        logger.info("Визуализация важности признаков...")
        visualize_feature_importance(model.feature_importance, top_n=30,
                                     output_path=os.path.join(output_dir, 'feature_importance.png'))

        # 19. Визуализация матрицы ошибок
        logger.info("Визуализация матрицы ошибок...")
        visualize_confusion_matrix(test_metrics['confusion_matrix'],
                                   output_path=os.path.join(output_dir, 'confusion_matrix.png'))

        # 20. Визуализация ROC-кривой
        logger.info("Визуализация ROC-кривой...")
        visualize_roc_curve(y_test, test_metrics['y_pred_proba'],
                            output_path=os.path.join(output_dir, 'roc_curve.png'))

        # 21. Визуализация PR-кривой
        logger.info("Визуализация PR-кривой...")
        visualize_pr_curve(y_test, test_metrics['y_pred_proba'],
                           output_path=os.path.join(output_dir, 'pr_curve.png'))

        # 22. Визуализация кривой калибровки
        logger.info("Визуализация кривой калибровки...")
        visualize_calibration_curve(y_test, test_metrics['y_pred_proba'],
                                    output_path=os.path.join(output_dir, 'calibration_curve.png'))

        # 23. Визуализация кривой обучения
        logger.info("Визуализация кривой обучения...")
        lgb_model = lightgbm.LGBMClassifier(
            random_state=random_state
        )

        # Передаем весь сбалансированный датасет для визуализации кривой обучения
        visualize_learning_curve(
            lgb_model,
            X_train_val_selected,
            y_train_val_balanced,
            cv=n_splits,
            output_path=os.path.join(output_dir, 'learning_curve.png'),
            random_state=random_state
        )

        # 24. Визуализация значений SHAP
        logger.info("Визуализация значений SHAP...")
        visualize_shap_values(model.model, X_test_selected,
                              n_samples=min(100, len(X_test_selected)),
                              output_dir=output_dir)

        # 25. Сохранение модели
        logger.info("Сохранение модели...")
        model_path = os.path.join(output_dir, 'tokenizator_model.txt')
        model.save_model(model_path)

        # Логирование артефактов модели
        for artifact in ['feature_importance.png', 'confusion_matrix.png', 'roc_curve.png',
                         'pr_curve.png', 'calibration_curve.png', 'learning_curve.png',
                         'shap_summary.png', 'lift_charts.png']:
            artifact_path = os.path.join(output_dir, artifact)
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

        # Логирование модели
        mlflow.lightgbm.log_model(model.model, "lightgbm_model")

        logger.info("Обучение и оценка модели успешно завершены!")

        return model, X_test_selected, y_test

    except Exception as e:
        logger.error(f"Ошибка в основном выполнении: {e}", exc_info=True)
        mlflow.log_param("error", str(e))
        raise

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Обучение и оценка модели классификации твитов')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_FILE, help='Путь к файлу с данными')
    parser.add_argument('--output', type=str, default='models', help='Директория для сохранения результатов')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Порог для бинаризации')
    parser.add_argument('--random_state', type=int, default=DEFAULT_RANDOM_STATE,
                        help='Seed для генератора случайных чисел')
    parser.add_argument('--n_splits', type=int, default=5, help='Количество фолдов для кросс-валидации')
    parser.add_argument('--test_size', type=float, default=0.2, help='Размер тестовой выборки (0-1)')

    args = parser.parse_args()

    logger.info("Запуск конвейера модели Tokenizator")

    try:
        # Создание директорий
        for dir_path in [args.output, 'feature_cache', 'plots']:
            os.makedirs(dir_path, exist_ok=True)

        # Запуск основного конвейера
        trained_model, X_test, y_test = train_tokenizator_model(
            args.data,
            args.output,
            args.threshold,
            args.random_state,
            args.n_splits,
            args.test_size
        )

        # Вывод сводки
        logger.info("Конвейер модели успешно завершен")
        logger.info(f"Точность модели: {trained_model.model_metrics['accuracy']:.4f}")
        logger.info(f"F1-score модели: {trained_model.model_metrics['f1']:.4f}")

    except Exception as e:
        logger.error(f"Ошибка в выполнении конвейера: {e}", exc_info=True)
        raise
