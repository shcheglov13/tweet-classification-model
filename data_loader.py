"""Функции для загрузки данных из различных источников"""

import os
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Загрузка данных из JSON-файла

    Args:
        file_path: Путь к JSON-файлу с данными

    Returns:
        pd.DataFrame: Загруженный датасет
    """
    logger.info(f"Загрузка данных из {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"Загружено {len(df)} твитов")
        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        raise


def save_dataframe(df: pd.DataFrame, file_path: str, format: str = 'csv') -> None:
    """
    Сохранение DataFrame в файл различных форматов

    Args:
        df: DataFrame для сохранения
        file_path: Путь к файлу
        format: Формат файла ('csv', 'json', 'parquet', 'pickle')
    """
    try:
        # Создание директории, если она не существует
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Сохранение в выбранном формате
        if format.lower() == 'csv':
            df.to_csv(file_path, index=True)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records')
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, index=True)
        elif format.lower() == 'pickle' or format.lower() == 'pkl':
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {format}")

        logger.info(f"DataFrame сохранен в {file_path} в формате {format}")
    except Exception as e:
        logger.error(f"Ошибка сохранения DataFrame: {e}")
        raise


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Загрузка DataFrame из файла

    Args:
        file_path: Путь к файлу

    Returns:
        pd.DataFrame: Загруженный DataFrame
    """
    try:
        # Определение формата файла по расширению
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.csv':
            return pd.read_csv(file_path, index_col=0)
        elif file_ext == '.json':
            return pd.read_json(file_path, orient='records')
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        elif file_ext in ['.pkl', '.pickle']:
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")

    except Exception as e:
        logger.error(f"Ошибка загрузки DataFrame: {e}")
        raise
