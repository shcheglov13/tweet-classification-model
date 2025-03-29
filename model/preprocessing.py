"""Функции для предобработки данных перед обучением модели"""

import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def scale_features(X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Масштабирование признаков с сохранением NaN значений на их местах

    Args:
        X: DataFrame с признаками
        scaler: Предварительно настроенный масштабировщик (опционально)

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Масштабированные признаки и масштабировщик
    """
    logger.info("Масштабирование признаков с сохранением NaN значений...")

    # Сохраняем маску NaN значений до масштабирования
    nan_mask = X.isna()

    # Временно заполняем NaN, чтобы StandardScaler мог работать
    # Используем 0, но только для масштабирования
    X_temp = X.fillna(0)

    # Масштабирование
    if scaler is None:
        scaler = StandardScaler()
        X_scaled_values = scaler.fit_transform(X_temp)
    else:
        X_scaled_values = scaler.transform(X_temp)

    # Создаем новый DataFrame из масштабированных значений
    X_scaled = pd.DataFrame(
        X_scaled_values,
        columns=X.columns,
        index=X.index
    )

    # Восстанавливаем NaN значения в местах, где они были изначально
    for col in X.columns:
        X_scaled.loc[nan_mask[col], col] = np.nan

    logger.info(f"Признаки масштабированы: {X_scaled.shape}")
    logger.info(f"Сохранено {nan_mask.sum().sum()} NaN значений")

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

def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None,
                    threshold: float = 100.0,
                    scaler: Optional[StandardScaler] = None) -> Tuple[
    pd.DataFrame, Optional[pd.Series], StandardScaler]:
    """
    Полная предобработка данных включая масштабирование признаков и бинаризацию цели.
    Сохраняет NaN значения в признаках для использования с LightGBM.

    Args:
        X: DataFrame с признаками
        y: Серия значений целевой переменной (опционально)
        threshold: Порог для бинаризации
        scaler: Предварительно настроенный масштабировщик (опционально)

    Returns:
        Tuple: Обработанные признаки с сохраненными NaN, бинаризованная цель (если предоставлена) и масштабировщик
    """
    logger.info(f"Начало предобработки данных с порогом {threshold}...")

    # Подсчет NaN до масштабирования
    initial_nan_count = X.isna().sum().sum()
    logger.info(f"Исходные данные содержат {initial_nan_count} NaN значений")

    # Проверка, есть ли колонки, содержащие только NaN
    nan_columns = X.columns[X.isna().all()].tolist()
    if nan_columns:
        logger.warning(f"Обнаружены колонки, содержащие только NaN: {nan_columns}")
        # Заполняем их нулями, иначе они могут вызвать проблемы
        X = X.copy()
        X[nan_columns] = X[nan_columns].fillna(0)
        logger.info(f"Колонки с только NaN заполнены нулями для стабильности обработки")

    # Масштабирование признаков с сохранением NaN
    X_scaled, scaler = scale_features(X, scaler)

    # Бинаризация целевой переменной, если она предоставлена
    y_binary = None
    if y is not None:
        y_binary = binarize_target(y, threshold)

    # Проверка, что NaN сохранились
    final_nan_count = X_scaled.isna().sum().sum()
    logger.info(f"После предобработки сохранено {final_nan_count} NaN значений")

    logger.info("Предобработка данных завершена")

    return X_scaled, y_binary, scaler