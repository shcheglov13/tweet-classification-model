"""Модуль для извлечения текстовых признаков из твитов"""

import os
import re
import emoji
import numpy as np
import pandas as pd
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Any, Union

from .base_extractor import BaseExtractor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BERTWEET_MODEL_NAME, TEXT_PCA_COMPONENTS, BATCH_SIZE, MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)


class TextFeatureExtractor(BaseExtractor):
    """
    Класс для извлечения текстовых признаков из твитов с использованием BERTweet
    """

    def __init__(self, device=None, cache_dir='feature_cache'):
        """
        Инициализация экстрактора текстовых признаков

        Args:
            device: Устройство для вычислений ('cpu' или 'cuda')
            cache_dir: Директория для кеширования
        """
        super().__init__(device, cache_dir)

        # Загрузка модели BERTweet
        logger.info("Загрузка модели BERTweet...")
        try:
            self.bertweet_tokenizer = AutoTokenizer.from_pretrained(BERTWEET_MODEL_NAME)
            self.bertweet_model = AutoModel.from_pretrained(BERTWEET_MODEL_NAME).to(self.device)
            self.bertweet_model.eval()  # Установка в режим оценки
            logger.info("Модель BERTweet успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели BERTweet: {e}")
            raise

        # Инициализация редуктора размерности
        self.text_reducer = None

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение текстовых признаков из данных твитов

        Args:
            df: DataFrame с данными твитов

        Returns:
            pd.DataFrame: DataFrame с текстовыми признаками
        """
        # Проверка наличия признаков в кеше
        cached_features = self.load_from_cache('text')
        if cached_features is not None:
            return cached_features

        logger.info("Извлечение текстовых признаков...")

        # Инициализация пустых DataFrame для признаков
        bertweet_df = pd.DataFrame(index=df.index)
        basic_metrics_df = pd.DataFrame(index=df.index)
        informal_slang_df = pd.DataFrame(index=df.index)

        # Размер пакета для обработки
        batch_size = BATCH_SIZE

        # Извлечение базовых текстовых метрик
        self._extract_basic_metrics(df, basic_metrics_df)

        # Извлечение эмбеддингов BERTweet
        all_embeddings = self._extract_bertweet_embeddings(df)

        # Применение PCA для снижения размерности
        reduced_embeddings = self._reduce_embeddings_dimension(all_embeddings)

        # Сохранение уменьшенных эмбеддингов
        for i, idx in enumerate(df.index):
            for j, emb in enumerate(reduced_embeddings[i]):
                bertweet_df.at[idx, f'bertweet_emb_{j}'] = emb

        # Извлечение признаков неформального языка
        self._extract_informal_language_features(df, informal_slang_df)

        # Объединение всех текстовых признаков
        text_features_df = pd.concat([bertweet_df, basic_metrics_df, informal_slang_df], axis=1)

        # Сохранение в кеш
        self.save_to_cache(text_features_df, 'text')

        logger.info(f"Извлечено {text_features_df.shape[1]} текстовых признаков")
        return text_features_df

    def _extract_basic_metrics(self, df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
        """
        Извлечение базовых текстовых метрик

        Args:
            df: DataFrame с данными твитов
            metrics_df: DataFrame для сохранения метрик
        """
        logger.info("Извлечение базовых текстовых метрик...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Сбор текстовых метрик"):
            # Получение текстового содержимого
            text = row['text'] or ""
            quoted_text = row['quoted_text'] or ""
            combined_text = text + " " + quoted_text

            # Длины текста
            metrics_df.at[idx, 'text_length'] = len(text)
            metrics_df.at[idx, 'quoted_text_length'] = len(quoted_text)
            metrics_df.at[idx, 'combined_text_length'] = len(combined_text)

            # Количество слов
            text_words = text.split() if text else []
            quoted_words = quoted_text.split() if quoted_text else []
            metrics_df.at[idx, 'text_word_count'] = len(text_words)
            metrics_df.at[idx, 'quoted_text_word_count'] = len(quoted_words)
            metrics_df.at[idx, 'combined_word_count'] = len(text_words) + len(quoted_words)

            # Средняя длина слова
            if text_words:
                metrics_df.at[idx, 'avg_word_length_text'] = sum(len(word) for word in text_words) / len(text_words)
            else:
                metrics_df.at[idx, 'avg_word_length_text'] = 0

            if quoted_words:
                metrics_df.at[idx, 'avg_word_length_quoted'] = sum(len(word) for word in quoted_words) / len(
                    quoted_words)
            else:
                metrics_df.at[idx, 'avg_word_length_quoted'] = 0

            # Специальные элементы
            hashtags = re.findall(r'#\w+', combined_text)
            mentions = re.findall(r'@\w+', combined_text)
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                              combined_text)
            emojis = [c for c in combined_text if emoji.is_emoji(c)]

            metrics_df.at[idx, 'hashtag_count'] = len(hashtags)
            metrics_df.at[idx, 'mention_count'] = len(mentions)
            metrics_df.at[idx, 'url_count'] = len(urls)
            metrics_df.at[idx, 'emoji_count'] = len(emojis)

            # Расчет плотностей
            combined_length = len(combined_text) if combined_text else 1  # Избегаем деления на ноль
            metrics_df.at[idx, 'hashtag_density'] = len(hashtags) / combined_length
            metrics_df.at[idx, 'mention_density'] = len(mentions) / combined_length
            metrics_df.at[idx, 'url_density'] = len(urls) / combined_length
            metrics_df.at[idx, 'emoji_density'] = len(emojis) / combined_length

    def _extract_bertweet_embeddings(self, df: pd.DataFrame) -> List[np.ndarray]:
        """
        Извлечение эмбеддингов BERTweet для всех твитов

        Args:
            df: DataFrame с данными твитов

        Returns:
            List[np.ndarray]: Список эмбеддингов для всех твитов
        """
        logger.info("Извлечение эмбеддингов BERTweet...")

        # Сбор всех текстов
        all_texts = []
        for idx, row in df.iterrows():
            text = row['text'] or ""
            quoted_text = row['quoted_text'] or ""
            combined_text = text + " " + quoted_text
            all_texts.append(combined_text)

        # Сбор эмбеддингов в пакетном режиме
        all_embeddings = []
        batch_size = BATCH_SIZE
        max_length = MAX_TEXT_LENGTH

        logger.info(f"Сбор эмбеддингов BERTweet с размером пакета {batch_size} и max_length={max_length}...")
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Сбор эмбеддингов текста"):
            batch_texts = all_texts[i:i + batch_size]

            try:
                # Обрабатываем только непустые тексты
                valid_indices = [j for j, txt in enumerate(batch_texts) if txt.strip()]
                if not valid_indices:
                    # Если все тексты в пакете пустые, добавляем нулевые эмбеддинги
                    for _ in range(len(batch_texts)):
                        all_embeddings.append(np.zeros(768))
                    continue

                valid_texts = [batch_texts[j] for j in valid_indices]

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

                # Добавляем эмбеддинги в общий список (с учетом пустых текстов)
                emb_idx = 0
                for j in range(len(batch_texts)):
                    if j in valid_indices:
                        all_embeddings.append(valid_embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        all_embeddings.append(np.zeros(768))

                # Очистка памяти после каждого пакета
                del inputs, outputs
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Ошибка извлечения эмбеддингов для пакета {i}: {e}")
                # Заполняем нулями в случае ошибки
                for _ in range(len(batch_texts)):
                    all_embeddings.append(np.zeros(768))

        return all_embeddings

    def _reduce_embeddings_dimension(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Снижение размерности эмбеддингов с использованием PCA

        Args:
            embeddings: Список эмбеддингов

        Returns:
            np.ndarray: Массив уменьшенных эмбеддингов
        """
        logger.info(f"Применение PCA для снижения размерности {len(embeddings)} текстовых эмбеддингов")
        self.text_reducer = PCA(n_components=TEXT_PCA_COMPONENTS, random_state=42)
        reduced_embeddings = self.text_reducer.fit_transform(np.array(embeddings))
        logger.info(f"Снижена размерность эмбеддингов BERTweet с 768 до {reduced_embeddings.shape[1]}")

        return reduced_embeddings

    def _extract_informal_language_features(self, df: pd.DataFrame, slang_df: pd.DataFrame) -> None:
        """
        Извлечение признаков неформального языка

        Args:
            df: DataFrame с данными твитов
            slang_df: DataFrame для сохранения признаков сленга
        """
        logger.info("Извлечение признаков неформального языка...")

        batch_size = BATCH_SIZE
        max_length = MAX_TEXT_LENGTH

        for i in tqdm(range(0, len(df), batch_size), desc="Обработка признаков неформального сленга"):
            batch_indices = df.index[i:i + batch_size]

            for idx in batch_indices:
                row = df.loc[idx]
                text = row['text'] or ""
                quoted_text = row['quoted_text'] or ""
                combined_text = text + " " + quoted_text

                # Соотношение заглавных букв
                if combined_text:
                    uppercase_chars = sum(1 for c in combined_text if c.isupper())
                    slang_df.at[idx, 'uppercase_ratio'] = uppercase_chars / len(combined_text)
                else:
                    slang_df.at[idx, 'uppercase_ratio'] = 0

                # Удлинение слов (например, "sooooo")
                elongation_pattern = r'(\w)\1{2,}'
                elongations = re.findall(elongation_pattern, combined_text)
                slang_df.at[idx, 'word_elongation_count'] = len(elongations)

                # Избыточная пунктуация
                punctuation_pattern = r'[!?]{2,}'
                excessive_punct = re.findall(punctuation_pattern, combined_text)
                slang_df.at[idx, 'excessive_punctuation_count'] = len(excessive_punct)

                # Перплексия BERTweet (использование вариации токенов как прокси)
                if combined_text:
                    try:
                        tokens = self.bertweet_tokenizer(
                            combined_text,
                            return_tensors="pt",
                            max_length=max_length,
                            padding="max_length",
                            truncation=True
                        ).to(self.device)

                        with torch.no_grad():
                            outputs = self.bertweet_model(**tokens, output_hidden_states=True)
                            # Использование вариации эмбеддингов как прокси для перплексии
                            last_hidden = outputs.last_hidden_state.squeeze()
                            if last_hidden.dim() > 1:  # Убедимся, что у нас есть 2D-тензор
                                token_variance = torch.var(last_hidden, dim=1).mean().item()
                            else:
                                token_variance = 0.0

                        slang_df.at[idx, 'perplexity_score'] = token_variance

                        # Очистка памяти
                        del tokens, outputs, last_hidden
                    except Exception as e:
                        logger.warning(f"Ошибка при расчете перплексии для индекса {idx}: {e}")
                        slang_df.at[idx, 'perplexity_score'] = 0
                else:
                    slang_df.at[idx, 'perplexity_score'] = 0

            # Очистка памяти после обработки пакета
            torch.cuda.empty_cache()
