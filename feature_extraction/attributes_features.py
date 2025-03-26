"""Модуль для извлечения признаков на основе атрибутов из разметки"""

import os

import numpy as np
import pandas as pd
import logging

from .base_extractor import BaseExtractor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class AttributesFeatureExtractor(BaseExtractor):
    """
    Класс для извлечения признаков на основе атрибутов из разметки (EM, SS, CT)

    Attributes:
        em_attributes (list): Список атрибутов эмоционального тона (EM1-EM11)
        ss_attributes (list): Список атрибутов стилистики речи (SS1-SS7)
        ct_attributes (list): Список атрибутов тем контента (CT1-CT8)
        all_attributes (list): Список всех атрибутов
    """

    def __init__(self, device=None, cache_dir='feature_cache'):
        """
        Инициализация экстрактора признаков атрибутов

        Args:
            device: Устройство для вычислений ('cpu' или 'cuda')
            cache_dir: Директория для кеширования
        """
        super().__init__(device, cache_dir)

        # Определение групп атрибутов
        self.em_attributes = [f"EM{i}" for i in range(1, 12)]
        self.ss_attributes = [f"SS{i}" for i in range(1, 8)]
        self.ct_attributes = [f"CT{i}" for i in range(1, 9)]

        # Все атрибуты
        self.all_attributes = self.em_attributes + self.ss_attributes + self.ct_attributes

        logger.info(f"Инициализирован экстрактор признаков атрибутов с {len(self.all_attributes)} атрибутами")

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из атрибутов из размеченных данных

        Args:
            df: DataFrame с данными твитов, содержащий столбец 'attributes' с атрибутами

        Returns:
            pd.DataFrame: DataFrame с признаками атрибутов
        """
        # Проверка наличия признаков в кеше
        cached_features = self.load_from_cache('attributes')
        if cached_features is not None:
            return cached_features

        logger.info("Извлечение признаков из атрибутов...")

        # Создание пустого DataFrame для признаков
        attributes_df = pd.DataFrame(index=df.index)

        # Проверка наличия столбца с атрибутами
        if 'attributes' not in df.columns:
            logger.warning("Столбец 'attributes' не найден в данных. Заполняем нулевыми значениями.")
            # Создаем пустой словарь для каждой строки
            df = df.copy()
            df['attributes'] = df.apply(lambda _: {}, axis=1)

        # Извлечение значений атрибутов
        for attr in self.all_attributes:
            attributes_df[attr] = df.apply(lambda row: self._get_attribute_value(row, attr), axis=1)

        # Сохранение в кеш
        self.save_to_cache(attributes_df, 'attributes')

        logger.info(f"Извлечено {attributes_df.shape[1]} признаков атрибутов")
        return attributes_df

    def _get_attribute_value(self, row, attribute):
        """
        Получение значения атрибута из строки DataFrame

        Args:
            row: Строка DataFrame
            attribute: Имя атрибута

        Returns:
            float: Значение атрибута
        """
        if isinstance(row['attributes'], dict) and attribute in row['attributes']:
            return row['attributes'][attribute]
        return 0.0
