# visualization/feature_viz.py
"""Функции для визуализации признаков и их важности"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


def visualize_feature_importance(feature_importance: pd.DataFrame,
                                 top_n: int = 20,
                                 output_path: str = 'feature_importance.png') -> None:
    """
    Визуализация топ-N наиболее важных признаков

    Args:
        feature_importance: DataFrame с важностью признаков
        top_n: Количество признаков для отображения
        output_path: Путь для сохранения изображения
    """
    logger.info(f"Визуализация топ-{top_n} важных признаков")

    # Получение топ-N признаков
    top_features = feature_importance.head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.xlabel('Важность (прирост)')
    plt.ylabel('Признак')
    plt.title(f'Топ-{top_n} важных признаков')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Визуализация важности признаков сохранена в '{output_path}'")


def visualize_feature_groups(feature_groups: Dict[str, List[str]],
                             feature_importance: Optional[pd.DataFrame] = None,
                             output_path: str = 'feature_groups.png') -> None:
    """
    Визуализация групп признаков и их размеров

    Args:
        feature_groups: Словарь с группами признаков
        feature_importance: DataFrame с важностью признаков (опционально)
        output_path: Путь для сохранения изображения
    """
    logger.info("Визуализация групп признаков")

    # Сбор данных о размерах групп
    group_sizes = {group: len(features) for group, features in feature_groups.items()}

    # Сбор данных о средней важности признаков в группах, если доступно
    group_importance = {}
    if feature_importance is not None:
        # Создание словаря важности признаков
        importance_dict = dict(zip(feature_importance['feature'], feature_importance['importance']))

        for group, features in feature_groups.items():
            # Фильтрация признаков, которые есть в словаре важности
            valid_features = [f for f in features if f in importance_dict]
            if valid_features:
                group_importance[group] = sum(importance_dict[f] for f in valid_features) / len(valid_features)
            else:
                group_importance[group] = 0

    # Визуализация размеров групп
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    groups = list(group_sizes.keys())
    sizes = [group_sizes[g] for g in groups]
    plt.bar(groups, sizes, color='skyblue')
    plt.xlabel('Группа признаков')
    plt.ylabel('Количество признаков')
    plt.title('Размеры групп признаков')
    plt.xticks(rotation=45, ha='right')

    # Визуализация средней важности групп, если доступно
    if feature_importance is not None:
        plt.subplot(1, 2, 2)
        importances = [group_importance[g] for g in groups]
        plt.bar(groups, importances, color='lightgreen')
        plt.xlabel('Группа признаков')
        plt.ylabel('Средняя важность')
        plt.title('Средняя важность признаков по группам')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Визуализация групп признаков сохранена в '{output_path}'")


def visualize_correlation_matrix(X: pd.DataFrame,
                                 top_n: int = 20,
                                 output_path: str = 'correlation_matrix.png') -> None:
    """
    Визуализация матрицы корреляции признаков

    Args:
        X: DataFrame с признаками
        top_n: Количество наиболее коррелирующих признаков для отображения
        output_path: Путь для сохранения изображения
    """
    logger.info(f"Визуализация матрицы корреляции для топ-{top_n} признаков")

    # Расчет корреляционной матрицы
    corr_matrix = X.corr().abs()

    # Выбор топ-N признаков с наибольшей корреляцией
    # Вычисляем среднюю корреляцию каждого признака со всеми остальными
    mean_corr = corr_matrix.mean()
    top_corr_features = mean_corr.nlargest(top_n).index

    # Создание матрицы корреляции для топ-N признаков
    top_corr_matrix = corr_matrix.loc[top_corr_features, top_corr_features]

    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f'Матрица корреляции для топ-{top_n} признаков')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Визуализация матрицы корреляции сохранена в '{output_path}'")
