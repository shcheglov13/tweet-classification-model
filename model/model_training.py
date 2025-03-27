"""Функции для обучения модели классификации твитов"""

import numpy as np
import pandas as pd
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Dict, List, Optional, Any
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc

logger = logging.getLogger(__name__)


def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42) -> Tuple:
    """
    Разделение данных на тестовую выборку и кроссвалидационные разбиения
    обучающей+валидационной выборки

    Args:
        X: DataFrame с признаками
        y: Серия целевых значений
        n_splits: Количество фолдов для кросс-валидации
        test_size: Размер тестовой выборки (0-1)
        random_state: Seed для генератора случайных чисел

    Returns:
        Tuple: X_train_val, X_test, y_train_val, y_test, kfold (StratifiedKFold object)
    """
    logger.info(f"Разделение данных на тестовую выборку и {n_splits}-фолдовую кросс-валидацию")

    # Проверка минимального размера классов
    min_class_size = y.value_counts().min()
    if min_class_size < n_splits:
        logger.warning(f"Минимальный размер класса ({min_class_size}) меньше количества фолдов ({n_splits}). "
                     f"Уменьшаем количество фолдов до {min_class_size}.")
        n_splits = min_class_size

    # Разделение на train+val и test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Инициализация стратифицированной кросс-валидации
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    logger.info(f"Обучающая+валидационная выборка: {X_train_val.shape}, Тестовая выборка: {X_test.shape}")
    logger.info(f"Используется стратифицированная {n_splits}-фолдовая кросс-валидация")

    return X_train_val, X_test, y_train_val, y_test, kfold


def train_model(
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series,
        kfold,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        optimization_metric: str = 'pr_auc') -> Tuple[lgb.Booster, pd.DataFrame, Dict]:
    """
    Обучение модели LightGBM с использованием кросс-валидации

    Args:
        X_train_val: DataFrame с признаками обучающей+валидационной выборки
        y_train_val: Серия целевых значений обучающей+валидационной выборки
        kfold: Объект кросс-валидации (StratifiedKFold)
        params: Параметры модели LightGBM
        feature_names: Имена признаков
        threshold: Порог для классификации
        optimization_metric: Метрика для оптимизации

    Returns:
        Tuple[lgb.Booster, pd.DataFrame, Dict]: Лучшая модель, важность признаков, результаты по фолдам
    """
    logger.info(f"Обучение модели LightGBM с кросс-валидацией, порогом {threshold} и метрикой {optimization_metric}")

    # Параметры по умолчанию, если не указаны
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_child_samples': 20,
            'max_depth': 8,
            'random_state': 42,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'gpu_use_dp': False,
            'max_bin': 63
        }

    # Использование имен признаков, если предоставлены
    if feature_names is None:
        feature_names = X_train_val.columns.tolist()

    # Подготовка для хранения результатов по фолдам
    fold_results = {}
    best_model = None
    best_val_metric = float('-inf')  # Для метрик типа AUC, F1, где выше - лучше
    combined_importance = pd.DataFrame({'feature': feature_names, 'importance': 0.0})

    # Обучение модели на каждом фолде
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
        logger.info(f"Обучение на фолде {fold_idx + 1}")

        # Разделение данных для текущего фолда
        X_train_fold = X_train_val.iloc[train_idx]
        y_train_fold = y_train_val.iloc[train_idx]
        X_val_fold = X_train_val.iloc[val_idx]
        y_val_fold = y_train_val.iloc[val_idx]

        # Создание объектов датасета
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold, feature_name=feature_names)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, feature_name=feature_names, reference=train_data)

        # Обучение модели
        callbacks = [
            lgb.early_stopping(50)
        ]

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=callbacks
        )

        # Получение важности признаков для текущего фолда
        fold_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance(importance_type='gain')
        })

        # Обновление общей важности признаков
        combined_importance['importance'] += fold_importance['importance'] / kfold.n_splits

        # Оценка модели на валидационной выборке
        y_pred_proba = model.predict(X_val_fold)
        y_pred_binary = (y_pred_proba > threshold).astype(int)

        # Расчет метрик с использованием переданного порога
        precision, recall, _ = precision_recall_curve(y_val_fold, y_pred_proba)
        pr_auc = auc(recall, precision)

        f1 = f1_score(y_val_fold, y_pred_binary)
        precision_val = precision_score(y_val_fold, y_pred_binary)
        recall_val = recall_score(y_val_fold, y_pred_binary)
        roc_auc = roc_auc_score(y_val_fold, y_pred_proba)

        metrics = {
            'f1': f1,
            'precision': precision_val,
            'recall': recall_val,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc
        }

        # Сохранение результатов фолда
        fold_results[fold_idx] = {
            'metrics': metrics,
            'importance': fold_importance
        }

        # Логирование метрик в зависимости от метрики оптимизации
        if optimization_metric == 'pr_auc':
            logger.info(f"Фолд {fold_idx + 1}: PR-AUC = {pr_auc:.4f}")
        elif optimization_metric == 'f1':
            logger.info(f"Фолд {fold_idx + 1}: F1 = {f1:.4f}")
        elif optimization_metric == 'roc_auc':
            logger.info(f"Фолд {fold_idx + 1}: ROC-AUC = {roc_auc:.4f}")
        else:
            logger.info(f"Фолд {fold_idx + 1}: F1 = {f1:.4f}, PR-AUC = {pr_auc:.4f}, ROC-AUC = {roc_auc:.4f}")

        # Проверка, является ли текущая модель лучшей по выбранной метрике
        current_metric = metrics.get(optimization_metric, f1)  # По умолчанию F1 если метрика не найдена
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_model = model

    # Сортировка важности признаков
    combined_importance = combined_importance.sort_values('importance', ascending=False)

    logger.info(f"Кросс-валидация завершена. Лучший {optimization_metric}: {best_val_metric:.4f}")

    return best_model, combined_importance, fold_results


def find_optimal_threshold(
        model: Any, X_val: pd.DataFrame, y_val: pd.Series,
        threshold_metric: str = 'f1',
        calibrator: Optional[Any] = None) -> float:
    """
    Поиск оптимального порога классификации для калиброванных вероятностей

    Args:
        model: Обученная модель
        X_val: DataFrame с признаками валидационной выборки
        y_val: Серия целевых значений валидационной выборки
        threshold_metric: Метрика для оптимизации порога ('f1', 'precision', 'recall')
        calibrator: Объект калибратора вероятностей (опционально)

    Returns:
        float: Оптимальный порог
    """
    logger.info(f"Поиск оптимального порога классификации с метрикой {threshold_metric}")

    # Получение предсказаний с учетом калибровки, если доступна
    if calibrator is not None and hasattr(calibrator, 'is_calibrated') and calibrator.is_calibrated:
        y_pred_proba = calibrator.predict_proba(X_val)
        logger.info("Используем калиброванные вероятности для поиска порога")
    else:
        y_pred_proba = model.predict(X_val)
        logger.info("Используем некалиброванные вероятности для поиска порога")

    # Поиск оптимального порога с использованием выбранной метрики
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)

        # Выбор метрики для оптимизации порога
        if threshold_metric == 'f1':
            score = f1_score(y_val, y_pred)
        elif threshold_metric == 'precision':
            score = precision_score(y_val, y_pred)
        elif threshold_metric == 'recall':
            score = recall_score(y_val, y_pred)
        else:
            # По умолчанию используем F1
            score = f1_score(y_val, y_pred)

        scores.append(score)

    # Поиск порога с наивысшим значением метрики
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]

    logger.info(f"Оптимальный порог: {best_threshold:.4f} с {threshold_metric}: {scores[best_idx]:.4f}")

    return best_threshold


def predict(model: lgb.Booster, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Предсказание на основе модели LightGBM

    Args:
        model: Обученная модель LightGBM
        X: DataFrame с признаками
        threshold: Порог классификации

    Returns:
        Tuple[np.ndarray, np.ndarray]: Бинарные предсказания и вероятности
    """
    logger.info(f"Выполнение предсказаний с порогом {threshold}")

    # Получение предсказаний
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > threshold).astype(int)

    return y_pred, y_pred_proba
