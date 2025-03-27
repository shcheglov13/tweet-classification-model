"""
Основной скрипт для обучения и оценки модели классификации твитов
"""

import os
import argparse
import mlflow
from typing import Tuple
import lightgbm
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data_loader import load_data
from feature_extraction.feature_manager import FeatureExtractor
from model.tokenizator_model import TokenizatorModel
from visualization.feature_viz import visualize_feature_importance
from visualization.performance_viz import (visualize_confusion_matrix, visualize_roc_curve,
                                           visualize_pr_curve, visualize_calibration_curve,
                                           visualize_learning_curve, visualize_lift,
                                           visualize_calibration)
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
        test_size: float = 0.2,
        optimization_metric: str = 'pr_auc',
        threshold_metric: str = 'f1') -> Tuple[TokenizatorModel, pd.DataFrame, pd.Series]:
    """
    Обучение и оценка модели классификации твитов

    Args:
        data_file: Путь к файлу с данными
        output_dir: Директория для сохранения результатов
        threshold: Порог для бинаризации
        random_state: Seed для генератора случайных чисел
        n_splits: Количество фолдов для кросс-валидации
        test_size: Размер тестовой выборки (0-1)
        optimization_metric: Метрика для оптимизации гиперпараметров ('pr_auc', 'f1', 'roc_auc')
        threshold_metric: Метрика для определения порога ('f1', 'precision', 'recall')

    Returns:
        Tuple: Обученная модель, тестовые признаки и тестовые целевые значения
    """
    # Создание директории для результатов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Запуск отслеживания MLflow
    mlflow.start_run(run_name="tokenizator_model")

    try:
        logger.info(f"Загрузка датасета из {data_file}...")
        df = load_data(data_file)

        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("positive_ratio", (df['total_trade_volume'] >= threshold).mean())
        mlflow.log_param("optimization_metric", optimization_metric)
        mlflow.log_param("threshold_metric", threshold_metric)

        logger.info("Извлечение признаков...")
        feature_extractor = FeatureExtractor()
        features_df, invalid_indices = feature_extractor.extract_all_features(df)

        if invalid_indices:
            logger.info(f"Удаление {len(invalid_indices)} невалидных постов с недоступными изображениями")
            valid_df = df.drop(invalid_indices)
            valid_features_df = features_df.drop(invalid_indices)
        else:
            valid_df = df
            valid_features_df = features_df

        logger.info(
            f"Инициализация модели с метрикой оптимизации '{optimization_metric}' и метрикой порога '{threshold_metric}'...")
        model = TokenizatorModel(
            random_state=random_state,
            optimization_metric=optimization_metric,
            threshold_metric=threshold_metric
        )

        logger.info("Предобработка данных...")
        X = valid_features_df.copy()
        y = valid_df['total_trade_volume']
        X, y_binary = model.preprocess_data(X, y, threshold=threshold)

        logger.info("Удаление сильно коррелирующих признаков...")
        X_reduced, dropped_features = model.analyze_feature_correlations(X, threshold=config.CORRELATION_THRESHOLD)

        logger.info("Группировка признаков по типу...")
        feature_groups = model.group_features(X_reduced)

        logger.info(f"Разделение данных с {n_splits}-фолдовой кросс-валидацией...")
        X_train_val, X_test, y_train_val, y_test, kfold = model.split_data(
            X_reduced, y_binary, n_splits=n_splits, test_size=test_size)

        # Анализ распределения классов
        class_stats = model.analyze_class_distribution(y_train_val)
        mlflow.log_params({
            'imbalance_ratio': class_stats['imbalance_ratio'],
            'uniformity_index': class_stats['uniformity_index']
        })

        # Определение оптимального метода обработки дисбаланса
        best_imbalance_method, imbalance_params = model.select_optimal_imbalance_method(
            X_train_val, y_train_val, kfold)
        mlflow.log_param('imbalance_method', best_imbalance_method)

        # Обработка дисбаланса с использованием лучшего метода
        X_train_val_balanced, y_train_val_balanced, _ = model.handle_class_imbalance(
            X_train_val, y_train_val, method=best_imbalance_method)

        logger.info(f"Оптимизация параметров модели по метрике {optimization_metric}")
        logger.info(
            f"Запуск совместной оптимизации гиперпараметров и отбора признаков по метрике {optimization_metric}...")
        best_params, selected_features = model.optimize_jointly(
            X_train_val_balanced, y_train_val_balanced,
            cv_outer=3, cv_inner=5, n_trials=30,
            optimization_metric=optimization_metric
        )

        logger.info("Отбор признаков с использованием оптимальных параметров...")
        X_train_val_selected, selected_features = model.select_features_with_optimal_parameters(
            X_train_val_balanced, y_train_val_balanced,
            params=best_params, trial_budget=50
        )

        # Обновляем тестовый набор с учетом выбранных признаков
        X_test_selected = X_test[selected_features]

        logger.info(
            f"Обучение модели с выбранными признаками и оптимальными гиперпараметрами по метрике {optimization_metric}...")
        model.train(
            X_train_val_selected, y_train_val_balanced, kfold,
            params=best_params, optimization_metric=optimization_metric
        )

        # Анализ стабильности результатов между фолдами
        logger.info("Анализ стабильности результатов между фолдами...")
        stability_metrics = model.analyze_fold_stability()

        logger.info("Переобучение финальной модели на всех тренировочных данных...")
        model.train_final_model(
            X_train_val_selected,
            y_train_val_balanced,
            params=best_params
        )

        logger.info(f"Оценка модели на тестовой выборке после оптимизация параметров по метрике {optimization_metric} (без калибровки)...")
        test_metrics_before_calibration = model.evaluate(X_test_selected, y_test, threshold=0.5)

        mlflow.log_metrics({
            'stage1_accuracy': test_metrics_before_calibration['accuracy'],
            'stage1_precision': test_metrics_before_calibration['precision'],
            'stage1_recall': test_metrics_before_calibration['recall'],
            'stage1_f1': test_metrics_before_calibration['f1'],
            'stage1_roc_auc': test_metrics_before_calibration['roc_auc'],
            'stage1_pr_auc': test_metrics_before_calibration['pr_auc']
        })

        # Сохранение визуализации кривой калибровки до калибровки
        visualize_calibration_curve(
            y_test,
            test_metrics_before_calibration['y_pred_proba'],
            output_path=os.path.join(output_dir, 'calibration_curve_before.png')
        )

        # Разделение обучающих данных для калибровки и проверки
        logger.info("Калибровка вероятностей с помощью Platt scaling")
        X_calib, X_calib_test, y_calib, y_calib_test = train_test_split(
            X_train_val_selected, y_train_val_balanced,
            test_size=0.3, random_state=random_state, stratify=y_train_val_balanced
        )

        # Выполнение калибровки вероятностей
        calibration_results = model.calibrate_probabilities(
            X_calib, y_calib,
            method='sigmoid', cv=5,
            output_dir=output_dir
        )

        mlflow.log_metrics({
            'brier_score_before': calibration_results.get('brier_score_before', 0),
            'brier_score_after': calibration_results.get('brier_score_after', 0),
            'brier_improvement': calibration_results.get('brier_improvement', 0),
            'log_loss_before': calibration_results.get('log_loss_before', 0),
            'log_loss_after': calibration_results.get('log_loss_after', 0),
            'log_loss_improvement': calibration_results.get('log_loss_improvement', 0)
        })

        logger.info("Оценка модели на тестовой выборке после калибровки (порог 0.5)...")
        test_metrics_after_calibration = model.evaluate(X_test_selected, y_test, threshold=0.5)
        mlflow.log_metrics({
            'stage2_accuracy': test_metrics_after_calibration['accuracy'],
            'stage2_precision': test_metrics_after_calibration['precision'],
            'stage2_recall': test_metrics_after_calibration['recall'],
            'stage2_f1': test_metrics_after_calibration['f1'],
            'stage2_roc_auc': test_metrics_after_calibration['roc_auc'],
            'stage2_pr_auc': test_metrics_after_calibration['pr_auc']
        })

        # Визуализация сравнения кривых калибровки до и после калибровки
        if model.calibrator.is_calibrated and 'y_pred_proba_calibrated' in test_metrics_after_calibration:
            visualize_calibration(
                y_test,
                test_metrics_before_calibration['y_pred_proba'],
                test_metrics_after_calibration['y_pred_proba_calibrated'],
                output_path=os.path.join(output_dir, 'calibration_comparison.png'),
                title='Сравнение кривых калибровки'
            )

        logger.info(f"Поиск оптимального порога для максимизации {threshold_metric}")

        # Поиск оптимального порога для откалиброванных вероятностей
        optimal_threshold = model.find_optimal_threshold(
            X_calib_test, y_calib_test,
            threshold_metric=threshold_metric
        )

        mlflow.log_param('optimal_threshold', optimal_threshold)

        logger.info(f"Итоговая оценка модели после поиска оптимального порога (с калибровкой, порог {optimal_threshold:.4f})...")
        final_test_metrics = model.evaluate(X_test_selected, y_test, threshold=optimal_threshold)
        mlflow.log_metrics({
            'final_accuracy': final_test_metrics['accuracy'],
            'final_precision': final_test_metrics['precision'],
            'final_recall': final_test_metrics['recall'],
            'final_f1': final_test_metrics['f1'],
            'final_roc_auc': final_test_metrics['roc_auc'],
            'final_pr_auc': final_test_metrics['pr_auc']
        })

        logger.info("Расчет и визуализация лифта...")
        bin_metrics = model.compute_lift(X_test_selected, y_test, bins=10)
        visualize_lift(bin_metrics, output_path=os.path.join(output_dir, 'lift_charts.png'))

        logger.info("Визуализация важности признаков...")
        visualize_feature_importance(
            model.feature_importance,
            top_n=30,
            output_path=os.path.join(output_dir, 'feature_importance.png')
        )

        logger.info("Визуализация матрицы ошибок...")
        visualize_confusion_matrix(
            final_test_metrics['confusion_matrix'],
            output_path=os.path.join(output_dir, 'confusion_matrix.png')
        )

        logger.info("Визуализация ROC-кривой и PR-кривой...")
        y_pred_proba = final_test_metrics.get('y_pred_proba_calibrated', final_test_metrics['y_pred_proba'])
        visualize_roc_curve(y_test, y_pred_proba, output_path=os.path.join(output_dir, 'roc_curve.png'))
        visualize_pr_curve(y_test, y_pred_proba, output_path=os.path.join(output_dir, 'pr_curve.png'))

        logger.info("Визуализация кривой обучения...")
        final_params = model.best_params.copy()
        visualize_learning_curve(
            lightgbm.LGBMClassifier(**final_params),
            X_train_val_selected,
            y_train_val_balanced,
            cv=n_splits,
            output_path=os.path.join(output_dir, 'learning_curve.png'),
            random_state=random_state,
            final_params=final_params
        )

        logger.info("Визуализация значений SHAP...")
        visualize_shap_values(
            model.model,
            X_test_selected,
            n_samples=min(100, len(X_test_selected)),
            output_dir=output_dir
        )

        logger.info("Сохранение итоговой модели...")
        model_path = os.path.join(output_dir, 'tokenizator_model.txt')
        model.save_model(model_path)

        for artifact in [
            'feature_importance.png', 'confusion_matrix.png', 'roc_curve.png',
            'pr_curve.png', 'calibration_curve_before.png', 'calibration_comparison.png',
            'learning_curve.png', 'lift_charts.png', 'shap_summary.png', 'shap_detailed.png',
            'shap_top_features.png'
        ]:
            artifact_path = os.path.join(output_dir, artifact)
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

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
    parser = argparse.ArgumentParser(description='Обучение и оценка модели классификации твитов')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_FILE, help='Путь к файлу с данными')
    parser.add_argument('--output', type=str, default='models', help='Директория для сохранения результатов')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Порог для бинаризации')
    parser.add_argument('--random_state', type=int, default=DEFAULT_RANDOM_STATE,
                        help='Seed для генератора случайных чисел')
    parser.add_argument('--n_splits', type=int, default=5, help='Количество фолдов для кросс-валидации')
    parser.add_argument('--test_size', type=float, default=0.2, help='Размер тестовой выборки (0-1)')
    parser.add_argument('--optimization_metric', type=str, default='pr_auc',
                        choices=['pr_auc', 'f1', 'roc_auc'],
                        help='Метрика для оптимизации гиперпараметров')
    parser.add_argument('--threshold_metric', type=str, default='f1',
                        choices=['f1', 'precision', 'recall'],
                        help='Метрика для определения порога')

    args = parser.parse_args()

    logger.info("Запуск конвейера модели tweet-classifier-model")
    logger.info(f"Метрика оптимизации гиперпараметров: {args.optimization_metric}")
    logger.info(f"Метрика определения порога: {args.threshold_metric}")

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
            args.test_size,
            args.optimization_metric,
            args.threshold_metric
        )

        logger.info("Конвейер модели успешно завершен")

    except Exception as e:
        logger.error(f"Ошибка в выполнении конвейера: {e}", exc_info=True)
        raise
