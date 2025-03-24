# model/preprocessing.py
"""Функции для предобработки данных перед обучением модели"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def scale_features(X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Масштабирование признаков

    Args:
        X: DataFrame с признаками
        scaler: Предварительно настроенный масштабировщик (опционально)

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Масштабированные признаки и масштабировщик
    """
    logger.info("Масштабирование признаков...")

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

    logger.info(f"Признаки масштабированы: {X_scaled.shape}")

    return X_scaled, scaler


def binarize_target(y: pd.Series, threshold: float = 100.0) -> pd.Series:
    """
    Бинаризация целевой переменной на основе порога

    Args:
        y: Серия значений целевой переменной
        threshold: Порог для бинаризации

    Returns:
        pd.Series: Бинаризованная целевая переменная
    """
    logger.info(f"Бинаризация целевой переменной с порогом {threshold}...")

    y_binary = (y >= threshold).astype(int)
    positive_ratio = y_binary.mean()

    logger.info(f"Целевая переменная бинаризована. Доля положительного класса: {positive_ratio:.4f}")

    return y_binary


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Обработка отсутствующих значений в DataFrame

    Args:
        df: DataFrame с данными
        strategy: Стратегия заполнения ('mean', 'median', 'zero')

    Returns:
        pd.DataFrame: DataFrame с заполненными отсутствующими значениями
    """
    logger.info(f"Обработка отсутствующих значений со стратегией '{strategy}'...")

    missing_count = df.isna().sum().sum()
    if missing_count == 0:
        logger.info("Отсутствующие значения не найдены")
        return df

    logger.info(f"Найдено {missing_count} отсутствующих значений")

    df_filled = df.copy()

    if strategy == 'mean':
        df_filled = df_filled.fillna(df_filled.mean())
    elif strategy == 'median':
        df_filled = df_filled.fillna(df_filled.median())
    elif strategy == 'zero':
        df_filled = df_filled.fillna(0)
    else:
        logger.warning(f"Неизвестная стратегия: {strategy}. Используется заполнение нулями.")
        df_filled = df_filled.fillna(0)

    return df_filled


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None,
                    threshold: float = 100.0,
                    scaler: Optional[StandardScaler] = None) -> Tuple[
    pd.DataFrame, Optional[pd.Series], StandardScaler]:
    """
    Полная предобработка данных включая масштабирование признаков и бинаризацию цели

    Args:
        X: DataFrame с признаками
        y: Серия значений целевой переменной (опционально)
        threshold: Порог для бинаризации
        scaler: Предварительно настроенный масштабировщик (опционально)

    Returns:
        Tuple: Обработанные признаки, бинаризованная цель (если предоставлена) и масштабировщик
    """
    logger.info(f"Начало предобработки данных с порогом {threshold}...")

    # Обработка отсутствующих значений в признаках
    X = handle_missing_values(X, strategy='zero')

    # Масштабирование признаков
    X_scaled, scaler = scale_features(X, scaler)

    # Бинаризация целевой переменной, если она предоставлена
    y_binary = None
    if y is not None:
        y_binary = binarize_target(y, threshold)

    logger.info("Предобработка данных завершена")

    return X_scaled, y_binary, scaler