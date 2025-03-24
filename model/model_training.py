# model/model_training.py
"""Функции для обучения модели классификации твитов"""

import numpy as np
import pandas as pd
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Any, Optional
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
               val_size: float = 0.25, random_state: int = 42) -> Tuple:
    """
    Разделение данных на обучающую, валидационную и тестовую выборки

    Args:
        X: DataFrame с признаками
        y: Серия целевых значений
        test_size: Размер тестовой выборки (0-1)
        val_size: Размер валидационной выборки от оставшихся данных (0-1)
        random_state: Seed для генератора случайных чисел

    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Разделение данных на обучающую, валидационную и тестовую выборки")

    ## Сначала разделяем на train+val и test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Затем разделяем train+val на train и validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )

    logger.info(
        f"Обучающая выборка: {X_train.shape}, Валидационная выборка: {X_val.shape}, Тестовая выборка: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              params: Optional[Dict] = None,
              feature_names: Optional[List[str]] = None) -> Tuple[lgb.Booster, pd.DataFrame]:
    """
    Обучение модели LightGBM

    Args:
        X_train: DataFrame с признаками обучающей выборки
        y_train: Серия целевых значений обучающей выборки
        X_val: DataFrame с признаками валидационной выборки
        y_val: Серия целевых значений валидационной выборки
        params: Параметры модели LightGBM
        feature_names: Имена признаков

    Returns:
        Tuple[lgb.Booster, pd.DataFrame]: Обученная модель и DataFrame с важностью признаков
    """
    logger.info("Обучение модели LightGBM")

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
        feature_names = X_train.columns.tolist()

    # Создание объектов датасета
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    # Обучение модели
    callbacks = [
        lgb.log_evaluation(100),
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

    # Получение важности признаков
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    logger.info(f"Модель обучена с лучшей итерацией: {model.best_iteration}")

    return model, feature_importance

def find_optimal_threshold(model: lgb.Booster, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Поиск оптимального порога классификации на основе F1-score

    Args:
        model: Обученная модель LightGBM
        X_val: DataFrame с признаками валидационной выборки
        y_val: Серия целевых значений валидационной выборки

    Returns:
        float: Оптимальный порог классификации
    """
    logger.info("Поиск оптимального порога классификации")


    # Получение предсказаний на валидационной выборке
    y_pred_val = model.predict(X_val)

    # Проверка различных порогов и вычисление F1-scores
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for threshold in thresholds:
        y_pred_binary = (y_pred_val > threshold).astype(int)
        f1 = f1_score(y_val, y_pred_binary)
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)

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