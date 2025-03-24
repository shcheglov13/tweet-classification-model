# model/feature_selection.py
"""Функции для анализа и отбора признаков"""

import numpy as np
import pandas as pd
import re
import logging
from typing import Tuple, List, Dict, Any, Optional
from sklearn.feature_selection import RFE, RFECV
import lightgbm as lgb

logger = logging.getLogger(__name__)


def analyze_correlations(X: pd.DataFrame, threshold: float = 0.9) -> Tuple[pd.DataFrame, List[str]]:
    """
    Анализ и удаление сильно коррелирующих признаков

    Args:
        X: DataFrame с признаками
        threshold: Порог корреляции для удаления

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame с выбранными признаками и список удаленных признаков
    """
    logger.info(f"Анализ корреляций признаков с порогом {threshold}")

    # Расчет корреляционной матрицы
    corr_matrix = X.corr().abs()

    # Извлечение верхнего треугольника корреляционной матрицы
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Поиск признаков с корреляцией выше порога
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    logger.info(f"Найдено {len(to_drop)} сильно коррелирующих признаков для удаления")

    # Удаление сильно коррелирующих признаков
    X_reduced = X.drop(to_drop, axis=1)

    return X_reduced, to_drop


def select_features_rfe(X: pd.DataFrame, y: pd.Series, k: int = 100,
                        random_state: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    """
    Выбор признаков с использованием Recursive Feature Elimination

    Args:
        X: DataFrame с признаками
        y: Серия целевых значений
        k: Количество признаков для выбора
        random_state: Seed для генератора случайных чисел

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame с выбранными признаками и список выбранных признаков
    """
    logger.info(f"Выбор топ-{k} признаков с использованием RFE")

    # Инициализация классификатора LightGBM
    estimator = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_estimators=100,
        random_state=random_state
    )

    # Инициализация RFE
    selector = RFE(
        estimator=estimator,
        n_features_to_select=k,
        step=10,
        verbose=1
    )

    # Обучение селектора
    selector.fit(X, y)

    # Получение выбранных признаков
    selected_features = X.columns[selector.support_].tolist()

    logger.info(f"Выбрано {len(selected_features)} признаков")

    # Создание новой матрицы признаков с выбранными признаками
    X_selected = X[selected_features]

    return X_selected, selected_features


def select_features_rfecv(X: pd.DataFrame, y: pd.Series,
                          random_state: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    """
    Выбор признаков с использованием RFECV с кросс-валидацией

    Args:
        X: DataFrame с признаками
        y: Серия целевых значений
        random_state: Seed для генератора случайных чисел

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame с выбранными признаками и список выбранных признаков
    """
    logger.info("Выбор признаков с использованием RFECV с кросс-валидацией")

    # Инициализация классификатора LightGBM
    estimator = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_estimators=100,
        random_state=random_state
    )

    # Инициализация RFECV
    selector = RFECV(
        estimator=estimator,
        step=10,
        cv=5,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )

    # Обучение селектора
    selector.fit(X, y)

    # Получение выбранных признаков
    selected_features = X.columns[selector.support_].tolist()

    logger.info(f"Выбрано {len(selected_features)} признаков")

    # Создание новой матрицы признаков с выбранными признаками
    X_selected = X[selected_features]

    return X_selected, selected_features


def group_features(X: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Группировка признаков по типу

    Args:
        X: DataFrame с признаками

    Returns:
        Dict[str, List[str]]: Словарь с группами признаков
    """
    logger.info("Группировка признаков по типу")

    # Определение шаблонов для идентификации групп признаков
    patterns = {
        'bertweet': 'bertweet_emb_',
        'text_metrics': 'text_|word_count|avg_word_length',
        'informal_slang': 'uppercase_ratio|word_elongation|excessive_punctuation|perplexity',
        'clip': 'clip_emb_',
        'emotional': 'emotion_|emotional_|coherence',
        'structural': 'tweet_type|has_|text_quoted_ratio|diversity',
        'temporal': 'hour|day_of_week|is_weekend'
    }

    # Группировка столбцов на основе шаблонов
    feature_groups = {}
    for group_name, pattern in patterns.items():
        columns = [col for col in X.columns if re.search(pattern, col)]
        feature_groups[group_name] = columns
        logger.info(f"Группа '{group_name}' имеет {len(columns)} признаков")

    # Проверка на неназначенные столбцы
    all_assigned = set()
    for cols in feature_groups.values():
        all_assigned.update(cols)

    unassigned = set(X.columns) - all_assigned
    if unassigned:
        logger.warning(f"Найдено {len(unassigned)} неназначенных признаков")
        feature_groups['other'] = list(unassigned)

    return feature_groups