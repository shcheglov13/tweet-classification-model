"""Функции для визуализации SHAP значений для интерпретации модели"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import shap
from typing import Any

logger = logging.getLogger(__name__)


def visualize_shap_values(model: Any, X_sample: pd.DataFrame, n_samples: int = 100,
                          output_dir: str = '.') -> None:
    """
    Визуализация значений SHAP для интерпретации модели

    Args:
        model: Обученная модель
        X_sample: DataFrame с примерами для анализа SHAP
        n_samples: Количество примеров для использования
        output_dir: Директория для сохранения изображений
    """
    logger.info(f"Визуализация значений SHAP (n_samples={n_samples})")

    try:
        # Использование подмножества данных для анализа SHAP
        if len(X_sample) > n_samples:
            X_subset = X_sample.sample(n_samples, random_state=42)
        else:
            X_subset = X_sample

        # Расчет значений SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_subset)

        # Сводный график
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_subset, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
        plt.close()

        # Детальный график
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_subset, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_detailed.png'))
        plt.close()

        # Генерация скрипичного графика для топ-20 признаков
        if X_subset.shape[1] > 10:
            # Получение важности признаков
            feature_importance = pd.DataFrame({
                'feature': X_subset.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)

            # Получение топ-20 признаков по важности
            top_features = feature_importance.head(20)['feature'].tolist()
            X_top = X_subset[top_features]

            plt.figure(figsize=(12, 10))
            shap.summary_plot(explainer.shap_values(X_top), X_top, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_top_features.png'))
            plt.close()

        logger.info("Визуализации SHAP сохранены в директории '{output_dir}'")

    except Exception as e:
        logger.error(f"Ошибка генерации визуализаций SHAP: {e}")