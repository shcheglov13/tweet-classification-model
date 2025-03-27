"""Модуль для извлечения структурных признаков из твитов"""

import os
import re

import numpy as np
import pandas as pd
import torch
import logging
from tqdm import tqdm
from typing import List

from .base_extractor import BaseExtractor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)


class StructuralFeatureExtractor(BaseExtractor):
    """
    Класс для извлечения структурных признаков из твитов
    """

    def __init__(self, device=None, cache_dir='feature_cache', bertweet_tokenizer=None, bertweet_model=None):
        """
        Инициализация экстрактора структурных признаков

        Args:
            device: Устройство для вычислений ('cpu' или 'cuda')
            cache_dir: Директория для кеширования
            bertweet_tokenizer: Токенизатор BERTweet (для повторного использования)
            bertweet_model: Модель BERTweet (для повторного использования)
        """
        super().__init__(device, cache_dir)

        # Сохранение ссылок на токенизатор и модель BERTweet, если предоставлены
        self.bertweet_tokenizer = bertweet_tokenizer
        self.bertweet_model = bertweet_model

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение структурных признаков из данных твитов

        Args:
            df: DataFrame с данными твитов

        Returns:
            pd.DataFrame: DataFrame со структурными признаками
        """
        # Проверка наличия признаков в кеше
        cached_features = self.load_from_cache('structural')
        if cached_features is not None:
            return cached_features

        logger.info("Извлечение структурных признаков...")

        # Инициализация пустого DataFrame для структурных признаков
        structural_df = pd.DataFrame(index=df.index)

        # 1. Признаки метаданных
        self._extract_metadata_features(df, structural_df)

        # 2. Признаки разнообразия текста
        if self.bertweet_tokenizer and self.bertweet_model:
            self._extract_text_diversity_features(df, structural_df)
        else:
            logger.warning("BERTweet модель не предоставлена, признаки разнообразия текста будут установлены на 0")
            structural_df['main_quoted_contrast'] = 0
            structural_df['internal_diversity'] = 0
            structural_df['combined_diversity'] = 0

        # 3. Временные признаки
        self._extract_temporal_features(df, structural_df)

        # Преобразование логических столбцов в int
        bool_columns = ['has_image', 'has_quoted_text']
        for col in bool_columns:
            if col in structural_df.columns:
                structural_df[col] = structural_df[col].astype(int)

        # Сохранение в кеш
        self.save_to_cache(structural_df, 'structural')

        logger.info(f"Извлечено {structural_df.shape[1]} структурных признаков")
        return structural_df

    def _extract_metadata_features(self, df: pd.DataFrame, structural_df: pd.DataFrame) -> None:
        """
        Извлечение признаков метаданных

        Args:
            df: DataFrame с данными твитов
            structural_df: DataFrame для сохранения структурных признаков
        """
        logger.info("Извлечение признаков метаданных...")

        # Тип твита (one-hot кодирование)
        tweet_types = pd.get_dummies(df['tweet_type'], prefix='tweet_type')
        for col in tweet_types.columns:
            structural_df[col] = tweet_types[col]

        # Определение типа медиа (one-hot кодирование)
        # Преобразование всех URL в строки, заменяя NaN на пустые строки
        image_urls = df['image_url'].fillna("").astype(str)

        # Признак наличия изображения
        has_image = image_urls != ""

        # Видео ссылки содержат "video" или "thumb" в пути
        video_pattern = re.compile(r'(video|tweet_video|amplify_video|video_thumb)', re.IGNORECASE)
        is_video = image_urls.apply(lambda url: bool(video_pattern.search(url)))

        # Тип медиа
        structural_df['media_type_only_text'] = ~has_image
        structural_df['media_type_video'] = has_image & is_video
        structural_df['media_type_image'] = has_image & ~is_video

        # Наличие цитируемого текста
        structural_df['has_quoted_text'] = df['quoted_text'].notna() & (df['quoted_text'] != "")

        # Соотношение длин текста (если существует цитируемый текст)
        for idx, row in df.iterrows():
            text_len = len(row['text']) if row['text'] else 0
            quoted_len = len(row['quoted_text']) if row['quoted_text'] else 0

            # Вычисляем отношение только если присутствуют оба текста
            if quoted_len > 0 and text_len > 0:
                structural_df.at[idx, 'text_quoted_ratio'] = text_len / quoted_len
            else:
                # Используем NaN как признак отсутствия отношения
                structural_df.at[idx, 'text_quoted_ratio'] = np.nan

    def _extract_text_diversity_features(self, df: pd.DataFrame, structural_df: pd.DataFrame) -> None:
        """
        Извлечение признаков разнообразия текста

        Args:
            df: DataFrame с данными твитов
            structural_df: DataFrame для сохранения структурных признаков
        """
        logger.info("Извлечение признаков разнообразия текста...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Вычисление разнообразия текста"):
            text = row['text'] or ""
            quoted_text = row['quoted_text'] or ""

            # Пропуск, если нет цитируемого текста
            if not quoted_text:
                structural_df.at[idx, 'main_quoted_contrast'] = 0
                structural_df.at[idx, 'internal_diversity'] = 0
                structural_df.at[idx, 'combined_diversity'] = 0
                continue

            # Получение эмбеддингов для основного и цитируемого текста
            main_embedding = self._get_bertweet_embeddings([text])
            quoted_embedding = self._get_bertweet_embeddings([quoted_text])

            # Расчет косинусного сходства (1 - сходство = контраст)
            if np.sum(main_embedding) != 0 and np.sum(quoted_embedding) != 0:
                similarity = np.dot(main_embedding.flatten(), quoted_embedding.flatten()) / (
                        np.linalg.norm(main_embedding) * np.linalg.norm(quoted_embedding)
                )
                contrast = 1 - similarity
            else:
                contrast = 0

            structural_df.at[idx, 'main_quoted_contrast'] = contrast

            # Внутреннее разнообразие (упрощенное)
            # Для полной реализации мы бы разделили тексты на предложения и сравнили их
            structural_df.at[idx, 'internal_diversity'] = min(contrast * 0.8, 1.0)

            # Комбинированная оценка разнообразия
            structural_df.at[idx, 'combined_diversity'] = (contrast + structural_df.at[idx, 'internal_diversity']) / 2

    def _extract_temporal_features(self, df: pd.DataFrame, structural_df: pd.DataFrame) -> None:
        """
        Извлечение временных признаков

        Args:
            df: DataFrame с данными твитов
            structural_df: DataFrame для сохранения структурных признаков
        """
        logger.info("Извлечение временных признаков...")

        for idx, row in df.iterrows():
            # Преобразование строки с датой в объект datetime
            if isinstance(row['created_at'], str):
                created_at = pd.to_datetime(row['created_at'])
            else:
                created_at = row['created_at']

            # Извлечение часа
            structural_df.at[idx, 'hour'] = created_at.hour

            # Извлечение дня недели (0-6, где 0=понедельник, 6=воскресенье)
            structural_df.at[idx, 'day_of_week'] = created_at.dayofweek

            # Определение, является ли день выходным (5=суббота, 6=воскресенье)
            structural_df.at[idx, 'is_weekend'] = 1 if created_at.dayofweek >= 5 else 0

    def _get_bertweet_embeddings(self, texts_list: List[str]) -> np.ndarray:
        """
        Получение эмбеддингов BERTweet для списка текстов

        Args:
            texts_list: Список текстов для получения эмбеддингов

        Returns:
            np.ndarray: Массив эмбеддингов
        """
        try:
            # Установка максимальной длины последовательности
            max_length = MAX_TEXT_LENGTH

            # Обрабатываем только непустые тексты
            valid_indices = [i for i, text in enumerate(texts_list) if text.strip()]

            # Если все тексты пустые, возвращаем нулевые эмбеддинги
            if not valid_indices:
                return np.zeros((len(texts_list), 768))

            valid_texts = [texts_list[i] for i in valid_indices]

            # Токенизация и подготовка для модели
            inputs = self.bertweet_tokenizer(
                valid_texts,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True
            ).to(self.device)

            # Получение эмбеддингов
            with torch.no_grad():
                outputs = self.bertweet_model(**inputs)
                # Получение эмбеддингов из последнего слоя, среднее объединение по токенам
                valid_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # Создаем массив для всех эмбеддингов (включая нулевые для пустых текстов)
            all_embeddings = np.zeros((len(texts_list), 768))

            # Заполняем эмбеддинги для непустых текстов
            for i, original_idx in enumerate(valid_indices):
                all_embeddings[original_idx] = valid_embeddings[i]

            # Очистка памяти
            del inputs, outputs
            torch.cuda.empty_cache()

            return all_embeddings

        except Exception as e:
            logger.error(f"Ошибка извлечения эмбеддингов BERTweet: {e}")
            # Возвращаем нули как резервный вариант
            return np.zeros((len(texts_list), 768))
