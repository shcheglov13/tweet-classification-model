"""Функции для обучения модели классификации твитов"""

import numpy as np
import pandas as pd
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import f1_score, precision_score, recall_score

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
        feature_names: Optional[List[str]] = None) -> Tuple[lgb.Booster, pd.DataFrame, Dict]:
    """
    Обучение модели LightGBM с использованием кросс-валидации

    Args:
        X_train_val: DataFrame с признаками обучающей+валидационной выборки
        y_train_val: Серия целевых значений обучающей+валидационной выборки
        kfold: Объект кросс-валидации (StratifiedKFold)
        params: Параметры модели LightGBM
        feature_names: Имена признаков

    Returns:
        Tuple[lgb.Booster, pd.DataFrame, Dict]: Лучшая модель, важность признаков, результаты по фолдам
    """
    logger.info("Обучение модели LightGBM с кросс-валидацией")

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
            # Настройки GPU
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
        y_pred = model.predict(X_val_fold)
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Расчет метрик
        metrics = {
            'f1': f1_score(y_val_fold, y_pred_binary),
            'precision': precision_score(y_val_fold, y_pred_binary),
            'recall': recall_score(y_val_fold, y_pred_binary)
        }

        # Сохранение результатов фолда
        fold_results[fold_idx] = {
            'metrics': metrics,
            'importance': fold_importance
        }

        logger.info(f"Фолд {fold_idx + 1}: F1 = {metrics['f1']:.4f}")

        # Проверка, является ли текущая модель лучшей
        if metrics['f1'] > best_val_metric:
            best_val_metric = metrics['f1']
            best_model = model

    # Сортировка важности признаков
    combined_importance = combined_importance.sort_values('importance', ascending=False)

    logger.info(f"Кросс-валидация завершена. Лучший F1: {best_val_metric:.4f}")

    return best_model, combined_importance, fold_results


def find_optimal_threshold(
        model: lgb.Booster,
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series,
        kfold) -> float:
    """
    Поиск оптимального порога классификации на основе F1-score с использованием кросс-валидации

    Args:
        model: Обученная модель LightGBM
        X_train_val: DataFrame с признаками обучающей+валидационной выборки
        y_train_val: Серия целевых значений обучающей+валидационной выборки
        kfold: Объект кросс-валидации

    Returns:
        float: Оптимальный порог классификации
    """
    logger.info("Поиск оптимального порога классификации с кросс-валидацией")

    # Инициализация массивов для хранения предсказаний и истинных значений
    all_preds = []
    all_true = []

    # Получение предсказаний на каждом фолде
    for train_idx, val_idx in kfold.split(X_train_val, y_train_val):
        # Разделение данных для текущего фолда
        X_val_fold = X_train_val.iloc[val_idx]
        y_val_fold = y_train_val.iloc[val_idx]

        # Получение предсказаний
        y_pred_val = model.predict(X_val_fold)

        # Сохранение предсказаний и истинных значений
        all_preds.extend(y_pred_val)
        all_true.extend(y_val_fold)

    # Преобразование в numpy массивы
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Проверка различных порогов и вычисление F1-scores
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for threshold in thresholds:
        y_pred_binary = (all_preds > threshold).astype(int)
        f1 = f1_score(all_true, y_pred_binary)
        precision = precision_score(all_true, y_pred_binary)
        recall = recall_score(all_true, y_pred_binary)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Поиск порога с наивысшим F1-score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    logger.info(f"Оптимальный порог: {best_threshold:.4f} с F1-score: {f1_scores[best_idx]:.4f}")

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
