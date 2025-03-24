# feature_extraction/visual_features.py
"""Модуль для извлечения визуальных признаков из изображений в твитах"""

import os
import hashlib
import numpy as np
import pandas as pd
import torch
import logging
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel

from utils.cache_utils import load_from_cache, save_to_cache
from .base_extractor import BaseExtractor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLIP_MODEL_NAME, IMAGE_CACHE_DIR, IMAGE_PCA_COMPONENTS

logger = logging.getLogger(__name__)


class VisualFeatureExtractor(BaseExtractor):
    """
    Класс для извлечения визуальных признаков из изображений в твитах с использованием CLIP
    """

    def __init__(self, device=None, cache_dir='feature_cache'):
        """
        Инициализация экстрактора визуальных признаков

        Args:
            device: Устройство для вычислений ('cpu' или 'cuda')
            cache_dir: Директория для кеширования
        """
        super().__init__(device, cache_dir)

        # Создание директории для кеширования изображений
        self.image_cache_dir = IMAGE_CACHE_DIR
        if not os.path.exists(self.image_cache_dir):
            os.makedirs(self.image_cache_dir)

        # Загрузка модели CLIP
        logger.info("Загрузка модели CLIP...")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=True)
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
            self.clip_model.eval()  # Установка в режим оценки
            logger.info("Модель CLIP успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели CLIP: {e}")
            raise

        # Инициализация редуктора размерности
        self.image_reducer = None

    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """
        Извлечение визуальных признаков из изображений в твитах

        Args:
            df: DataFrame с данными твитов

        Returns:
            Tuple[pd.DataFrame, List[int]]: DataFrame с визуальными признаками и список индексов невалидных постов
        """
        # Проверка наличия признаков в кеше
        cached_features = self.load_from_cache('visual')
        invalid_indices_file = os.path.join(self.cache_dir, 'visual_invalid_indices.pkl')

        if cached_features is not None and os.path.exists(invalid_indices_file):
            logger.info(f"Загрузка визуальных признаков из кеша: {self.get_cache_path('visual')}")
            invalid_indices = load_from_cache(invalid_indices_file)
            return cached_features, invalid_indices

        logger.info("Извлечение визуальных признаков...")

        # Инициализация пустого DataFrame для признаков
        clip_df = pd.DataFrame(index=df.index)

        # Список для хранения индексов невалидных постов
        invalid_indices = []

        # Добавление флага для твитов с изображениями
        has_image = df['image_url'].notna() & (df['image_url'] != "")
        clip_df['has_image'] = has_image.astype(int)

        # Обработка только при наличии изображений
        image_count = has_image.sum()
        if image_count > 0:
            logger.info(f"Найдено {image_count} твитов с изображениями для обработки")

            # Сбор всех CLIP эмбеддингов
            all_image_embeddings = []
            valid_indices = []

            # Обработка каждого твита с изображением
            for idx, row in tqdm(df[has_image].iterrows(), total=image_count, desc="Сбор эмбеддингов изображений"):
                try:
                    # Получение URL изображения
                    image_url = row['image_url']

                    # Получение эмбеддингов CLIP и статуса доступности
                    image_embeddings, is_valid = self._get_clip_embeddings(image_url)

                    # Если изображение недоступно, пометить пост как невалидный
                    if not is_valid:
                        invalid_indices.append(idx)
                        logger.warning(f"Изображение недоступно для поста {idx}, пост будет исключен из обучения")
                        continue

                    # Добавляем эмбеддинги для последующего снижения размерности
                    all_image_embeddings.append(image_embeddings)
                    valid_indices.append(idx)

                except Exception as e:
                    logger.error(f"Ошибка обработки изображения {row['image_url']}: {e}")
                    # Пометка поста как невалидного при ошибке обработки изображения
                    invalid_indices.append(idx)

            # Применяем PCA ко всем собранным эмбеддингам изображений
            if len(all_image_embeddings) > 1:
                logger.info(f"Применение PCA для снижения размерности {len(all_image_embeddings)} CLIP эмбеддингов")

                self.image_reducer = PCA(n_components=IMAGE_PCA_COMPONENTS, random_state=42)
                reduced_embeddings = self.image_reducer.fit_transform(np.array(all_image_embeddings))
                logger.info(f"Снижена размерность эмбеддингов CLIP с 768 до {reduced_embeddings.shape[1]}")

                # Сохраняем уменьшенные эмбеддинги
                for i, idx in enumerate(valid_indices):
                    for j, emb in enumerate(reduced_embeddings[i]):
                        clip_df.at[idx, f'clip_emb_{j}'] = emb

            elif len(all_image_embeddings) == 1:
                # Если только одно изображение, берем первые n значений
                logger.warning(
                    f"Только одно доступное изображение. Используем первые {IMAGE_PCA_COMPONENTS} значений оригинального эмбеддинга.")
                reduced_embedding = all_image_embeddings[0][:IMAGE_PCA_COMPONENTS]
                for j, emb in enumerate(reduced_embedding):
                    clip_df.at[valid_indices[0], f'clip_emb_{j}'] = emb

            else:
                logger.warning("Нет действительных изображений для обработки.")
        else:
            logger.info("Изображения не найдены в датасете, пропуск обработки CLIP")

        # Заполнение отсутствующих значений для твитов без изображений
        for i in range(IMAGE_PCA_COMPONENTS):  # Заполнение всех возможных столбцов эмбеддингов clip
            if f'clip_emb_{i}' in clip_df.columns:
                clip_df[f'clip_emb_{i}'] = clip_df[f'clip_emb_{i}'].fillna(0)
            else:
                clip_df[f'clip_emb_{i}'] = 0

        # Сохранение в кеш
        self.save_to_cache(clip_df, 'visual')
        save_to_cache(invalid_indices, invalid_indices_file)

        logger.info(f"Извлечено {clip_df.shape[1]} визуальных признаков")
        logger.info(f"Найдено {len(invalid_indices)} постов с недоступными изображениями")

        return clip_df, invalid_indices

    def _get_clip_embeddings(self, image_url: str) -> Tuple[np.ndarray, bool]:
        """
        Извлечение визуальных эмбеддингов CLIP для данного URL изображения

        Args:
            image_url: URL изображения

        Returns:
            Tuple[np.ndarray, bool]: Эмбеддинги и флаг успешного извлечения
        """
        try:
            # Создание имени файла кеша на основе хеша URL
            image_hash = hashlib.md5(image_url.encode()).hexdigest()
            cache_file = os.path.join(self.image_cache_dir, f"{image_hash}.jpg")

            # Проверка наличия изображения в кеше
            if os.path.exists(cache_file):
                img = Image.open(cache_file)
            else:
                # Скачивание и кеширование изображения
                response = requests.get(image_url, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Ошибка загрузки изображения с URL {image_url}: код состояния {response.status_code}")
                    return np.zeros(768), False  # Недоступное изображение

                img = Image.open(BytesIO(response.content))
                img.save(cache_file)  # Сохранение для будущего использования

            # Обработка изображения через CLIP
            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)

            # Получение эмбеддингов
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
                embeddings = outputs.cpu().numpy()[0]

            return embeddings, True  # Возврат эмбеддингов и флага успеха
        except Exception as e:
            logger.error(f"Ошибка получения CLIP эмбеддингов для {image_url}: {e}")
            # Возврат нулей и флага неудачи
            return np.zeros(768), False  # Типичный размер CLIP эмбеддинга