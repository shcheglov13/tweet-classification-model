"""Функции для сохранения и загрузки моделей и связанных объектов"""

import os
import pickle
import logging
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def save_model(model: Any,
               filepath: str,
               best_threshold: float = 0.5,
               feature_names: Optional[List[str]] = None,
               selected_features: Optional[List[str]] = None,
               feature_groups: Optional[Dict] = None,
               model_metrics: Optional[Dict] = None,
               scaler: Any = None,
               feature_importance: Optional[pd.DataFrame] = None,
               calibrator: Any = None,
               additional_info: Optional[Dict] = None) -> bool:
    """
    Сохраняет модель и связанные с ней объекты

    Args:
        model: Обученная модель LightGBM
        filepath: Путь для сохранения модели
        best_threshold: Оптимальный порог для модели
        feature_names: Имена признаков
        selected_features: Выбранные признаки
        feature_groups: Группы признаков
        model_metrics: Метрики модели
        scaler: Масштабировщик признаков
        feature_importance: DataFrame с важностью признаков
        calibrator: Калибратор вероятностей
        additional_info: Дополнительная информация о модели

    Returns:
        bool: Статус успешного сохранения
    """
    try:
        # Создание директории, если она не существует
        model_dir = os.path.dirname(filepath)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Сохранение модели LightGBM
        model.save_model(filepath)

        # Сохранение дополнительной информации о модели
        model_info = {
            'best_threshold': best_threshold,
            'feature_names': feature_names,
            'selected_features': selected_features,
            'feature_groups': feature_groups,
            'model_metrics': model_metrics,
            'additional_info': additional_info or {}
        }

        with open(f"{filepath.replace('.txt', '')}_info.pkl", 'wb') as f:
            pickle.dump(model_info, f)

        # Сохранение масштабировщика
        if scaler:
            with open(f"{filepath.replace('.txt', '')}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)

        # Сохранение важности признаков
        if feature_importance is not None:
            feature_importance.to_csv(f"{filepath.replace('.txt', '')}_importance.csv", index=False)

        # Сохранение калибратора
        if calibrator is not None:
            with open(f"{filepath.replace('.txt', '')}_calibrator.pkl", 'wb') as f:
                pickle.dump(calibrator, f)
            logger.info("Сохранен калибратор вероятностей")

        logger.info(f"Модель и связанные объекты сохранены в {filepath} и связанных файлах")

        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения модели: {e}")
        return False


def load_model(filepath: str) -> Tuple[Any, Dict, Any, Optional[pd.DataFrame], Optional[Any]]:
    """
    Загружает модель и связанные с ней объекты

    Args:
        filepath: Путь к сохраненной модели

    Returns:
        Tuple: (модель, информация о модели, масштабировщик, важность признаков, калибратор)
    """
    try:
        # Загрузка модели LightGBM
        if not os.path.exists(filepath):
            logger.error(f"Файл модели {filepath} не найден")
            return None, {}, None, None, None

        model = lgb.Booster(model_file=filepath)
        model_info = {}
        scaler = None
        feature_importance = None
        calibrator = None

        # Загрузка дополнительной информации о модели
        info_filepath = f"{filepath.replace('.txt', '')}_info.pkl"
        if os.path.exists(info_filepath):
            with open(info_filepath, 'rb') as f:
                model_info = pickle.load(f)

        # Загрузка масштабировщика
        scaler_filepath = f"{filepath.replace('.txt', '')}_scaler.pkl"
        if os.path.exists(scaler_filepath):
            with open(scaler_filepath, 'rb') as f:
                scaler = pickle.load(f)

        # Загрузка важности признаков, если доступно
        importance_filepath = f"{filepath.replace('.txt', '')}_importance.csv"
        if os.path.exists(importance_filepath):
            feature_importance = pd.read_csv(importance_filepath)

        # Загрузка калибратора, если доступно
        calibrator_filepath = f"{filepath.replace('.txt', '')}_calibrator.pkl"
        if os.path.exists(calibrator_filepath):
            with open(calibrator_filepath, 'rb') as f:
                calibrator = pickle.load(f)
            logger.info("Загружен калибратор вероятностей")

        logger.info(f"Модель и связанные объекты загружены из {filepath} и связанных файлов")

        return model, model_info, scaler, feature_importance, calibrator
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return None, {}, None, None, None
