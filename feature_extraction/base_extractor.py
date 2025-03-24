# feature_extraction/base_extractor.py
"""Базовый класс для всех экстракторов признаков"""

import os
import torch
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Union

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache_utils import ensure_cache_dir, cache_exists, save_to_cache, load_from_cache
from config import CACHE_DIR

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Базовый абстрактный класс для экстракторов признаков

    Attributes:
        cache_dir (str): Директория для кеширования извлеченных признаков
        device (str): Устройство для вычислений ('cpu' или 'cuda')
    """

    def __init__(self, device: Optional[str] = None, cache_dir: str = CACHE_DIR):
        """
        Инициализация базового экстрактора признаков

        Args:
            device: Устройство для вычислений ('cpu' или 'cuda')
            cache_dir: Директория для кеширования
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir

        # Создание директории кеша, если она не существует
        ensure_cache_dir(self.cache_dir)

        logger.info(f"Инициализирован экстрактор признаков с устройством: {self.device}")

    def get_cache_path(self, feature_type: str) -> str:
        """
        Получение пути к файлу кеша для определенного типа признаков

        Args:
            feature_type: Тип признаков

        Returns:
            str: Полный путь к файлу кеша
        """
        return os.path.join(self.cache_dir, f'{feature_type}_features.pkl')

    def load_from_cache(self, feature_type: str) -> Optional[pd.DataFrame]:
        """
        Загрузка признаков из кеша

        Args:
            feature_type: Тип признаков

        Returns:
            pd.DataFrame: DataFrame с признаками или None, если кеш не найден
        """
        cache_path = self.get_cache_path(feature_type)
        if cache_exists(cache_path):
            logger.info(f"Загрузка {feature_type} признаков из кеша: {cache_path}")
            return load_from_cache(cache_path)
        return None

    def save_to_cache(self, df: pd.DataFrame, feature_type: str) -> None:
        """
        Сохранение признаков в кеш

        Args:
            df: DataFrame с признаками
            feature_type: Тип признаков
        """
        cache_path = self.get_cache_path(feature_type)
        logger.info(f"Сохранение {feature_type} признаков в кеш: {cache_path}")
        save_to_cache(df, cache_path)

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Абстрактный метод извлечения признаков

        Args:
            df: DataFrame с данными

        Returns:
            pd.DataFrame: DataFrame с извлеченными признаками
        """
        pass