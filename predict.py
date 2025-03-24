"""
Скрипт для использования обученной модели для предсказаний на новых данных
"""

import argparse
import pandas as pd
from typing import Dict, Optional, Tuple
from model.tokenizator_model import TokenizatorModel
from feature_extraction.feature_manager import FeatureExtractor
from utils.logging_utils import setup_logger
from config import DEFAULT_MODEL_PATH, CACHE_DIR

logger = setup_logger()


def load_post_data(post_data: Dict) -> pd.DataFrame:
    """
    Загрузка данных поста в DataFrame

    Args:
        post_data: Словарь с данными поста

    Returns:
        pd.DataFrame: DataFrame с данными поста
    """
    return pd.DataFrame([post_data])


def predict_post_suitability(model_path: str, post_data: Dict, threshold: Optional[float] = None,
                             cache_dir: str = CACHE_DIR) -> Tuple[bool, float]:
    """
    Предсказание пригодности поста для токенизации

    Args:
        model_path: Путь к сохраненной модели
        post_data: Словарь с данными поста
        threshold: Порог для классификации (если None, используется лучший порог модели)
        cache_dir: Директория для кеширования

    Returns:
        Tuple[bool, float]: Флаг пригодности и вероятность пригодности
    """
    # Загрузка модели
    model = TokenizatorModel()
    if not model.load_model(model_path):
        logger.error(f"Не удалось загрузить модель из {model_path}")
        return False, 0.0

    # Преобразование данных поста в DataFrame
    post_df = load_post_data(post_data)

    # Извлечение признаков
    feature_extractor = FeatureExtractor(cache_dir=cache_dir)
    features_df, invalid_indices = feature_extractor.extract_all_features(post_df)

    # Проверка наличия недействительных изображений
    if invalid_indices and 0 in invalid_indices:
        logger.warning("Пост содержит недоступное изображение, считается непригодным")
        return False, 0.0

    # Выполнение предсказания
    y_pred, y_pred_proba = model.predict(features_df, threshold)

    return bool(y_pred[0]), float(y_pred_proba[0])


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Предсказание пригодности поста для токенизации')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Путь к модели')
    parser.add_argument('--threshold', type=float, help='Порог классификации')
    parser.add_argument('--text', type=str, required=True, help='Текст поста')
    parser.add_argument('--quoted_text', type=str, default='', help='Цитируемый текст')
    parser.add_argument('--image_url', type=str, default='', help='URL изображения')
    parser.add_argument('--tweet_type', type=str, default='SINGLE',
                        choices=['SINGLE', 'REPLY', 'QUOTE', 'RETWEET'], help='Тип твита')

    args = parser.parse_args()

    # Формирование данных поста
    post_data = {
        'text': args.text,
        'quoted_text': args.quoted_text,
        'image_url': args.image_url if args.image_url else None,
        'tweet_type': args.tweet_type,
        'created_at': pd.Timestamp.now()
    }

    # Предсказание пригодности
    is_suitable, probability = predict_post_suitability(
        args.model, post_data, args.threshold)

    # Вывод результата
    print(f"Пригодность для токенизации: {'Да' if is_suitable else 'Нет'}")
    print(f"Вероятность пригодности: {probability:.4f}")
