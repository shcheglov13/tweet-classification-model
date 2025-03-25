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

        # Логирование информации об извлечении признаков
        mlflow.log_param("total_features", valid_features_df.shape[1])
        mlflow.log_param("valid_posts", len(valid_df))

        # 3. Инициализация модели
        logger.info("Инициализация модели...")
        model = TokenizatorModel(random_state=random_state)

        # 4. Предобработка данных
        logger.info("Предобработка данных...")
        X = valid_features_df.copy()
        y = valid_df['total_trade_volume']
        X, y_binary = model.preprocess_data(X, y, threshold=threshold)

        # Логирование информации о предобработке
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("positive_class_ratio", y_binary.mean())

        # 5. Удаление сильно коррелирующих признаков
        logger.info("Удаление сильно коррелирующих признаков...")
        X_reduced, dropped_features = model.analyze_feature_correlations(X, threshold=0.9)

        # Логирование информации об анализе корреляций
        mlflow.log_param("initial_features", X.shape[1])
        mlflow.log_param("features_after_correlation", X_reduced.shape[1])
        mlflow.log_param("dropped_correlated_features", len(dropped_features))

        # 6. Группировка признаков по типу
        logger.info("Группировка признаков по типу...")
        feature_groups = model.group_features(X_reduced)

        # 7. Разделение данных с использованием кросс-валидации
        logger.info(f"Разделение данных с {n_splits}-фолдовой кросс-валидацией...")
        X_train_val, X_test, y_train_val, y_test, kfold = model.split_data(
            X_reduced, y_binary, n_splits=n_splits, test_size=test_size)

        # Логирование информации о разделении
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("train_val_size", X_train_val.shape[0])
        mlflow.log_param("test_size_actual", X_test.shape[0])

        # 8. Выбор признаков с использованием кросс-валидации
        logger.info("Выбор признаков с кросс-валидацией...")
        X_train_val_selected, selected_features = model.select_features(
            X_train_val, y_train_val, kfold, k=100)
        X_test_selected = X_test[selected_features]

        # Логирование информации о выборе признаков
        mlflow.log_param("selected_features_count", len(selected_features))

        # 9. Инкрементальная оценка признаков с кросс-валидацией
        logger.info("Инкрементальная оценка групп признаков с кросс-валидацией...")
        incremental_results = model.incremental_feature_evaluation(
            X_train_val, y_train_val, kfold)

        # 10. Обучение модели с кросс-валидацией
        logger.info("Обучение модели с кросс-валидацией...")
        fold_results = model.train(X_train_val_selected, y_train_val, kfold)

        # 11. Оптимизация гиперпараметров с кросс-валидацией
        logger.info("Оптимизация гиперпараметров с кросс-валидацией...")
        model.optimize_hyperparameters(X_train_val_selected, y_train_val, kfold, n_trials=30)

        # 12. Поиск оптимального порога с кросс-валидацией
        logger.info("Поиск оптимального порога с кросс-валидацией...")
        optimal_threshold = model.find_optimal_threshold(X_train_val_selected, y_train_val, kfold)

        # Логирование информации о пороге
        mlflow.log_param("optimal_threshold", optimal_threshold)

        # 13. Оценка модели на тестовой выборке
        logger.info("Оценка модели на тестовой выборке...")
        test_metrics = model.evaluate(X_test_selected, y_test, threshold=optimal_threshold)

        # Логирование метрик
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        # 14. Расчет и визуализация лифта
        logger.info("Расчет и визуализация лифта...")
        bin_metrics = model.compute_lift(X_test_selected, y_test, bins=10)
        visualize_lift(bin_metrics, output_path=os.path.join(output_dir, 'lift_charts.png'))

        # 15. Визуализация важности признаков
        logger.info("Визуализация важности признаков...")
        visualize_feature_importance(model.feature_importance, top_n=30,
                                     output_path=os.path.join(output_dir, 'feature_importance.png'))

        # 16. Визуализация матрицы ошибок
        logger.info("Визуализация матрицы ошибок...")
        visualize_confusion_matrix(test_metrics['confusion_matrix'],
                                   output_path=os.path.join(output_dir, 'confusion_matrix.png'))

        # 17. Визуализация ROC-кривой
        logger.info("Визуализация ROC-кривой...")
        visualize_roc_curve(y_test, test_metrics['y_pred_proba'],
                            output_path=os.path.join(output_dir, 'roc_curve.png'))

        # 18. Визуализация PR-кривой
        logger.info("Визуализация PR-кривой...")
        visualize_pr_curve(y_test, test_metrics['y_pred_proba'],
                           output_path=os.path.join(output_dir, 'pr_curve.png'))

        # 19. Визуализация кривой калибровки
        logger.info("Визуализация кривой калибровки...")
        visualize_calibration_curve(y_test, test_metrics['y_pred_proba'],
                                    output_path=os.path.join(output_dir, 'calibration_curve.png'))

        # 20. Визуализация кривой обучения (не требует изменений)
        logger.info("Визуализация кривой обучения...")
        lgb_model = lightgbm.LGBMClassifier(
            random_state=random_state,
            feature_name='auto'
        )
        # Используем произвольный фолд для визуализации кривой обучения
        train_idx, val_idx = next(kfold.split(X_train_val_selected, y_train_val))
        visualize_learning_curve(
            lgb_model,
            X_train_val_selected.iloc[train_idx], y_train_val.iloc[train_idx],
            output_path=os.path.join(output_dir, 'learning_curve.png')
        )

        # 21. Визуализация значений SHAP
        logger.info("Визуализация значений SHAP...")
        visualize_shap_values(model.model, X_test_selected,
                              n_samples=min(100, len(X_test_selected)),
                              output_dir=output_dir)

        # 22. Сохранение модели
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
        # Завершение отслеживания MLflow
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
