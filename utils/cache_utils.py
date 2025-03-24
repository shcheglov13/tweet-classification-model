"""Утилиты для кеширования данных и результатов"""

import os
import pickle
import pandas as pd
import hashlib
from typing import Any


def ensure_cache_dir(cache_dir: str) -> None:
    """Проверяет существование директории кеша и создает ее при необходимости"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


def cache_exists(cache_path: str) -> bool:
    """Проверяет существование кеша по указанному пути"""
    return os.path.exists(cache_path)


def save_to_cache(data: Any, cache_path: str) -> None:
    """Сохраняет данные в кеш"""
    # Создаем директорию для кеша, если она не существует
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Определяем формат сохранения по расширению
    if cache_path.endswith('.pkl'):
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    elif cache_path.endswith('.csv'):
        if isinstance(data, pd.DataFrame):
            data.to_csv(cache_path, index=True)
        else:
            raise TypeError("Data must be a pandas DataFrame for CSV caching")
    elif cache_path.endswith('.json'):
        if isinstance(data, pd.DataFrame):
            data.to_json(cache_path, orient='records')
        else:
            raise TypeError("Data must be a pandas DataFrame for JSON caching")
    elif cache_path.endswith('.parquet'):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path, index=True)
        else:
            raise TypeError("Data must be a pandas DataFrame for Parquet caching")
    else:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)


def load_from_cache(cache_path: str) -> Any:
    """Загружает данные из кеша"""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    # Определяем формат загрузки по расширению
    if cache_path.endswith('.pkl'):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    elif cache_path.endswith('.csv'):
        return pd.read_csv(cache_path, index_col=0)
    elif cache_path.endswith('.json'):
        return pd.read_json(cache_path, orient='records')
    elif cache_path.endswith('.parquet'):
        return pd.read_parquet(cache_path)
    else:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)


def generate_cache_key(data: Any, prefix: str = "") -> str:
    """Генерирует ключ кеша на основе данных"""
    if isinstance(data, str):
        hash_obj = hashlib.md5(data.encode())
    elif isinstance(data, pd.DataFrame):
        hash_obj = hashlib.md5(pd.util.hash_pandas_object(data).values)
    else:
        hash_obj = hashlib.md5(pickle.dumps(data))

    return f"{prefix}_{hash_obj.hexdigest()}" if prefix else hash_obj.hexdigest()


def cache_or_execute(func, cache_path: str, *args, **kwargs):
    """Выполняет функцию или возвращает результат из кеша"""
    if os.path.exists(cache_path):
        return load_from_cache(cache_path)

    result = func(*args, **kwargs)
    save_to_cache(result, cache_path)
    return result