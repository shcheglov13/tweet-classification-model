# model/evaluation.py
"""Функции для оценки модели и расчета различных метрик производительности"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, precision_recall_curve, auc,
                             confusion_matrix, classification_report, average_precision_score,
                             roc_curve)
import lightgbm as lgb

logger = logging.getLogger(__name__)


def evaluate_model(model: lgb.Booster, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict:
    """
    Оценка модели на заданных данных

    Args:
        model: Обученная модель LightGBM
        X: DataFrame с признаками
        y: Серия целевых значений
        threshold: Порог классификации

    Returns:
        Dict: Словарь с метриками производительности
    """
    logger.info(f"Оценка модели с порогом {threshold}")

    # Получение предсказаний
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > threshold).astype(int)

    # Расчет метрик
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Расчет PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    # Логирование метрик
    logger.info(f"Точность: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"PR AUC: {pr_auc:.4f}")

    # Расчет матрицы ошибок
    conf_mat = confusion_matrix(y, y_pred)
    logger.info(f"Матрица ошибок:\n{conf_mat}")

    # Детальный отчет о классификации
    class_report = classification_report(y, y_pred)
    logger.info(f"Отчет о классификации:\n{class_report}")

    # Возврат метрик в виде словаря
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': conf_mat,
        'classification_report': class_report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    return metrics


def compute_lift(model: lgb.Booster, X: pd.DataFrame, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    """
    Расчет и возврат данных для lift-диаграммы

    Args:
        model: Обученная модель LightGBM
        X: DataFrame с признаками
        y: Серия целевых значений
        bins: Количество бинов для диаграммы

    Returns:
        pd.DataFrame: DataFrame с данными для lift-диаграммы
    """
    logger.info(f"Расчет lift-диаграммы с {bins} бинами")

    # Получение предсказаний
    y_pred_proba = model.predict(X)

    # Создание DataFrame с фактическими и предсказанными значениями
    lift_df = pd.DataFrame({
        'actual': y,
        'predicted_proba': y_pred_proba
    })

    # Сортировка по предсказанной вероятности в порядке убывания
    lift_df = lift_df.sort_values('predicted_proba', ascending=False)

    # Разделение на равные бины
    lift_df['bin'] = pd.qcut(lift_df.index, bins, labels=False)

    # Расчет среднего фактического значения для каждого бина
    bin_metrics = lift_df.groupby('bin').agg({
        'actual': 'mean',
        'predicted_proba': 'mean'
    }).reset_index()

    # Расчет базовой линии (общее среднее)
    baseline = lift_df['actual'].mean()

    # Расчет лифта для каждого бина
    bin_metrics['lift'] = bin_metrics['actual'] / baseline

    # Расчет кумулятивного лифта
    bin_metrics['cumulative_lift'] = (lift_df.groupby('bin')['actual'].sum().cumsum() /
                                      (lift_df['actual'].sum() * (np.arange(1, bins + 1) / bins)))

    logger.info(f"Рассчитаны данные lift-диаграммы с базовой линией {baseline:.4f}")

    return bin_metrics