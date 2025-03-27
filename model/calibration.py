"""Функции для калибровки вероятностей модели"""

import numpy as np
import pandas as pd
import logging
import os
from typing import Tuple, Dict, Any, Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss

from visualization.performance_viz import visualize_calibration

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """
    Класс для калибровки вероятностей с помощью Platt scaling

    Attributes:
        calibrated_classifier: Калиброванный классификатор
        is_calibrated (bool): Флаг, указывающий, была ли выполнена калибровка
    """

    def __init__(self):
        """Инициализация калибратора"""
        self.calibrated_classifier = None
        self.is_calibrated = False

    def fit(self, model: Any, X_calibration: pd.DataFrame, y_calibration: pd.Series,
            method: str = 'sigmoid', cv: int = 5) -> 'PlattCalibrator':
        """
        Обучение калибратора вероятностей

        Args:
            model: Обученная модель LightGBM
            X_calibration: DataFrame с признаками для калибровки
            y_calibration: Серия целевых значений для калибровки
            method: Метод калибровки ('sigmoid' для Platt scaling)
            cv: Количество фолдов для кросс-валидации
        """
        logger.info(f"Калибровка вероятностей с методом '{method}' и {cv} фолдами")

        # Создание калибратора
        self.calibrated_classifier = CalibratedClassifierCV(
            model, method=method, cv=cv, n_jobs=-1, ensemble=True
        )

        # Обучение калибратора
        self.calibrated_classifier.fit(X_calibration, y_calibration)
        self.is_calibrated = True

        logger.info("Калибровка вероятностей завершена")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получение калиброванных вероятностей

        Args:
            X: DataFrame с признаками
        """
        if not self.is_calibrated:
            logger.warning("Калибратор не обучен. Возвращаются некалиброванные вероятности.")
            return np.zeros((len(X), 2))

        return self.calibrated_classifier.predict_proba(X)[:, 1]

def calibrate_model(model: Any, X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.3, random_state: int = 42,
                    calibration_method: str = 'sigmoid', cv: int = 5,
                    output_dir: Optional[str] = None) -> Tuple[PlattCalibrator, Dict]:
    """
    Калибровка вероятностей модели с помощью Platt scaling
    """
    logger.info("Запуск калибровки вероятностей модели")

    # Разделение данных на обучающую и тестовую выборки
    X_calib, X_test_calib, y_calib, y_test_calib = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Создание и обучение калибратора
    calibrator = PlattCalibrator()
    calibrator.fit(model, X_calib, y_calib, method=calibration_method, cv=cv)

    # Получение предсказаний до и после калибровки
    y_pred_proba_before = model.predict(X_test_calib)
    y_pred_proba_after = calibrator.predict_proba(X_test_calib)

    # Визуализация кривой калибровки, если указана директория
    if output_dir:
        output_path = os.path.join(output_dir, 'calibration_curve_comparison.png')
        visualize_calibration(
            y_test_calib,
            y_pred_proba_before,
            y_pred_proba_after,
            output_path=output_path,
            title='Сравнение кривых калибровки'
        )

    # Вычисление метрик калибровки
    # Среднеквадратичная ошибка калибровки (Brier score)
    brier_before = brier_score_loss(y_test_calib, y_pred_proba_before)
    brier_after = brier_score_loss(y_test_calib, y_pred_proba_after)

    # Логарифмическая функция потерь
    log_loss_before = log_loss(y_test_calib, y_pred_proba_before)
    log_loss_after = log_loss(y_test_calib, y_pred_proba_after)

    calibration_results = {
        'brier_score_before': brier_before,
        'brier_score_after': brier_after,
        'brier_improvement': (brier_before - brier_after) / brier_before * 100,
        'log_loss_before': log_loss_before,
        'log_loss_after': log_loss_after,
        'log_loss_improvement': (log_loss_before - log_loss_after) / log_loss_before * 100
    }

    logger.info(f"Brier score до калибровки: {brier_before:.6f}")
    logger.info(f"Brier score после калибровки: {brier_after:.6f}")
    logger.info(f"Улучшение Brier score: {calibration_results['brier_improvement']:.2f}%")

    logger.info(f"Log loss до калибровки: {log_loss_before:.6f}")
    logger.info(f"Log loss после калибровки: {log_loss_after:.6f}")
    logger.info(f"Улучшение log loss: {calibration_results['log_loss_improvement']:.2f}%")

    return calibrator, calibration_results