"""Функции для анализа и отбора признаков"""

import numpy as np
import pandas as pd
import re
import logging
from typing import Tuple, List, Dict
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


def select_features_from_model(
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series,
        kfold,
        random_state: int = 42,
        threshold: str = 'mean',
        k: int = 100) -> Tuple[pd.DataFrame, List[str]]:
    """
    Выбор признаков с использованием кросс-валидации

    Args:
        X_train_val: DataFrame с признаками обучающей+валидационной выборки
        y_train_val: Серия целевых значений обучающей+валидационной выборки
        kfold: Объект кросс-валидации
        random_state: Seed для генератора случайных чисел
        threshold: Порог для отбора признаков ('mean', 'median', или числовое значение)
        k: Максимальное количество признаков для выбора

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame с выбранными признаками и список выбранных признаков
    """
    logger.info(f"Выбор признаков с использованием кросс-валидации (threshold={threshold}, max_k={k})")

    # Инициализация словаря для хранения важности признаков по фолдам
    feature_importance_dict = {feature: 0.0 for feature in X_train_val.columns}

    # Обучение моделей на каждом фолде для определения важности признаков
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
        logger.info(f"Определение важности признаков на фолде {fold_idx + 1}")

        # Разделение данных для текущего фолда
        X_train_fold = X_train_val.iloc[train_idx]
        y_train_fold = y_train_val.iloc[train_idx]

        # Инициализация и обучение модели для определения важности признаков
        estimator = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            n_estimators=100,
            importance_type='gain',
            random_state=random_state,
            verbose=-1
        )

        estimator.fit(X_train_fold, y_train_fold)

        # Обновление словаря важности признаков
        for feature, importance in zip(X_train_val.columns, estimator.feature_importances_):
            feature_importance_dict[feature] += importance / kfold.n_splits

    # Создание DataFrame с важностью признаков
    feature_importance = pd.DataFrame({
        'feature': list(feature_importance_dict.keys()),
        'importance': list(feature_importance_dict.values())
    }).sort_values('importance', ascending=False)

    # Выбор топ-k признаков
    selected_features = feature_importance.head(k)['feature'].tolist()

    logger.info(f"Выбрано {len(selected_features)} признаков")

    # Создание новой матрицы признаков с выбранными признаками
    X_selected = X_train_val[selected_features]

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
        'emotional': 'emotion_|emotional_',
        'structural': 'tweet_type|has_image',
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
