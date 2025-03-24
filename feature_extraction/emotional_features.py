"""Модуль для извлечения эмоциональных признаков из твитов"""

import os
import numpy as np
import pandas as pd
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaForSequenceClassification

from .base_extractor import BaseExtractor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMOTION_MODEL_NAME, EMOTION_LABELS

logger = logging.getLogger(__name__)


class EmotionalFeatureExtractor(BaseExtractor):
    """
    Класс для извлечения эмоциональных признаков из твитов с использованием модели эмоций
    """

    def __init__(self, device=None, cache_dir='feature_cache'):
        """
        Инициализация экстрактора эмоциональных признаков

        Args:
            device: Устройство для вычислений ('cpu' или 'cuda')
            cache_dir: Директория для кеширования
        """
        super().__init__(device, cache_dir)

        # Загрузка модели эмоций
        logger.info("Загрузка модели эмоций...")
        try:
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
            self.emotion_model = RobertaForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME).to(self.device)
            self.emotion_model.eval()  # Установка в режим оценки

            # Метки эмоций
            self.emotion_labels = EMOTION_LABELS
            logger.info("Модель эмоций успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмоций: {e}")
            raise

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение эмоциональных признаков из данных твитов

        Args:
            df: DataFrame с данными твитов

        Returns:
            pd.DataFrame: DataFrame с эмоциональными признаками
        """
        # Проверка наличия признаков в кеше
        cached_features = self.load_from_cache('emotional')
        if cached_features is not None:
            return cached_features

        logger.info("Извлечение эмоциональных признаков...")

        # Инициализация пустого DataFrame для эмоциональных признаков
        emotional_df = pd.DataFrame(index=df.index)

        # Обработка каждого твита
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка эмоциональных признаков"):
            # Получение текстового содержимого
            text = row['text'] or ""
            quoted_text = row['quoted_text'] or ""
            combined_text = text + " " + quoted_text

            # Пропуск пустых текстов
            if not combined_text.strip():
                # Установка нулевых значений для эмоций
                for emotion in self.emotion_labels:
                    emotional_df.at[idx, f'emotion_{emotion}'] = 0
                emotional_df.at[idx, 'emotional_valence'] = 0
                emotional_df.at[idx, 'emotional_intensity'] = 0
                continue

            # Получение вероятностей эмоций
            emotion_probs = self._get_emotion_probabilities(combined_text)

            # Сохранение вероятностей эмоций
            for emotion, prob in zip(self.emotion_labels, emotion_probs):
                emotional_df.at[idx, f'emotion_{emotion}'] = prob

            # Расчет эмоциональной валентности (позитивные - негативные эмоции)
            positive_emotions = ['joy', 'love', 'optimism', 'trust', 'anticipation', 'surprise']
            negative_emotions = ['anger', 'disgust', 'fear', 'pessimism', 'sadness']

            positive_score = sum(emotional_df.at[idx, f'emotion_{e}'] for e in positive_emotions)
            negative_score = sum(emotional_df.at[idx, f'emotion_{e}'] for e in negative_emotions)

            # Нормализация в диапазоне [-1, 1]
            total = positive_score + negative_score
            if total > 0:
                emotional_df.at[idx, 'emotional_valence'] = (positive_score - negative_score) / total
            else:
                emotional_df.at[idx, 'emotional_valence'] = 0

            # Расчет эмоциональной интенсивности (сумма всех вероятностей эмоций)
            emotional_df.at[idx, 'emotional_intensity'] = sum(emotion_probs)

        # Сохранение в кеш
        self.save_to_cache(emotional_df, 'emotional')

        logger.info(f"Извлечено {emotional_df.shape[1]} эмоциональных признаков")
        return emotional_df

    def _get_emotion_probabilities(self, text: str) -> np.ndarray:
        """
        Получение вероятностей эмоций для заданного текста

        Args:
            text: Текст для анализа

        Returns:
            np.ndarray: Массив вероятностей эмоций
        """
        try:
            # Токенизация входных данных
            inputs = self.emotion_tokenizer(text, return_tensors="pt",
                                            truncation=True,
                                            max_length=128).to(self.device)

            # Получение предсказаний
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
                logits = outputs.logits

                # Применение сигмоиды для мультиклассовой классификации
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            return probs
        except Exception as e:
            logger.error(f"Ошибка получения вероятностей эмоций: {e}")
            return np.zeros(len(self.emotion_labels))
