# feature_extraction/feature_manager.py
"""Класс-оркестратор для управления извлечением признаков"""

import os
import torch
import logging
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional

from .text_features import TextFeatureExtractor
from .visual_features import VisualFeatureExtractor
from .emotional_features import EmotionalFeatureExtractor
from .structural_features import StructuralFeatureExtractor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CACHE_DIR

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Класс-оркестратор для управления извлечением всех типов признаков

    Attributes:
        cache_dir (str): Директория для кеширования признаков
        device (str): Устройство для вычислений ('cpu' или 'cuda')
        text_extractor: Экстрактор текстовых признаков
        visual_extractor: Экстрактор визуальных признаков
        emotional_extractor: Экстрактор эмоциональных признаков
        structural_extractor: Экстрактор структурных признаков
    """

    def __init__(self, device=None, cache_dir=CACHE_DIR):
        """
        Инициализация экстрактора признаков с моделями

        Args:
            device: Устройство для вычислений ('cpu' или 'cuda')
            cache_dir: Директория для кеширования признаков
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

        self.cache_dir = cache_dir
        logger.info(f"Используется устройство: {self.device}")

        # Создание директории кеша, если она не существует
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Инициализация экстракторов признаков
        logger.info("Инициализация экстракторов признаков...")

        # Инициализация текстового экстрактора
        self.text_extractor = TextFeatureExtractor(device=self.device, cache_dir=self.cache_dir)

        # Инициализация визуального экстрактора
        self.visual_extractor = VisualFeatureExtractor(device=self.device, cache_dir=self.cache_dir)

        # Инициализация эмоционального экстрактора
        self.emotional_extractor = EmotionalFeatureExtractor(device=self.device, cache_dir=self.cache_dir)

        # Инициализация структурного экстрактора с повторным использованием моделей
        self.structural_extractor = StructuralFeatureExtractor(
            device=self.device,
            cache_dir=self.cache_dir,
            bertweet_tokenizer=self.text_extractor.bertweet_tokenizer,
            bertweet_model=self.text_extractor.bertweet_model
        )

    def extract_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """
        Извлечение всех признаков из датасета

        Args:
            df: DataFrame с данными твитов

        Returns:
            Tuple[pd.DataFrame, List[int]]: DataFrame со всеми признаками и список индексов невалидных постов
        """
        logger.info("Запуск конвейера извлечения признаков")

        # Путь к файлу кеша
        cache_file = os.path.join(self.cache_dir, 'all_features.pkl')
        invalid_indices_file = os.path.join(self.cache_dir, 'invalid_indices.pkl')

        # Проверка наличия признаков в кеше
        if os.path.exists(cache_file) and os.path.exists(invalid_indices_file):
            logger.info(f"Загрузка признаков из кеша: {cache_file}")
            with open(invalid_indices_file, 'rb') as f:
                import pickle
                invalid_indices = pickle.load(f)
            return pd.read_pickle(cache_file), invalid_indices

        # Создание пустого DataFrame для хранения всех признаков
        features_df = pd.DataFrame(index=df.index)
        invalid_indices = []

        # 1. Извлечение текстовых признаков
        logger.info("Извлечение текстовых признаков...")
        text_features = self.text_extractor.extract(df)
        features_df = pd.concat([features_df, text_features], axis=1)

        # 2. Извлечение визуальных признаков
        logger.info("Извлечение визуальных признаков...")
        visual_features, img_invalid_indices = self.visual_extractor.extract(df)
        features_df = pd.concat([features_df, visual_features], axis=1)
        invalid_indices.extend(img_invalid_indices)

        # 3. Извлечение эмоциональных признаков
        logger.info("Извлечение эмоциональных признаков...")
        emotional_features = self.emotional_extractor.extract(df)
        features_df = pd.concat([features_df, emotional_features], axis=1)

        # 4. Извлечение структурных признаков
        logger.info("Извлечение структурных признаков...")
        structural_features = self.structural_extractor.extract(df)
        features_df = pd.concat([features_df, structural_features], axis=1)

        # Сохранение признаков в кеш
        features_df.to_pickle(cache_file)
        with open(invalid_indices_file, 'wb') as f:
            import pickle
            pickle.dump(invalid_indices, f)
        logger.info(f"Сохранены признаки в кеш: {cache_file}")

        logger.info(f"Извлечение признаков завершено. Всего признаков: {features_df.shape[1]}")
        logger.info(f"Найдено {len(invalid_indices)} невалидных постов, которые будут исключены из обучения")

        return features_df, invalid_indices