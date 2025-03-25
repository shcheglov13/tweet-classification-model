"""Функции для визуализации метрик производительности модели"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, StratifiedKFold
from typing import Any

logger = logging.getLogger(__name__)


def visualize_confusion_matrix(conf_matrix: np.ndarray,
                               output_path: str = 'confusion_matrix.png') -> None:
    """
    Визуализация матрицы ошибок

    Args:
        conf_matrix: Матрица ошибок
        output_path: Путь для сохранения изображения
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Не подходящие', 'Подходящие'],
                yticklabels=['Не подходящие', 'Подходящие'])
    plt.xlabel('Предсказанная метка')
    plt.ylabel('Истинная метка')
    plt.title('Матрица ошибок')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Визуализация матрицы ошибок сохранена в '{output_path}'")


def visualize_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                        output_path: str = 'roc_curve.png') -> None:
    """
    Визуализация ROC-кривой

    Args:
        y_true: Истинные целевые значения
        y_pred_proba: Предсказанные вероятности
        output_path: Путь для сохранения изображения
    """
    from sklearn.metrics import roc_curve, auc

    # Расчет ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Построение ROC-кривой
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (площадь = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Характеристическая кривая приёмника (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Визуализация ROC-кривой сохранена в '{output_path}'")


def visualize_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                       output_path: str = 'pr_curve.png') -> None:
    """
    Визуализация кривой Precision-Recall

    Args:
        y_true: Истинные целевые значения
        y_pred_proba: Предсказанные вероятности
        output_path: Путь для сохранения изображения
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score, auc

    # Расчет кривой Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    # Построение кривой Precision-Recall
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR-кривая (AP = {ap:.2f})')

    # Добавление базовой линии, представляющей долю положительных образцов
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--',
                label=f'Базовая линия (Доля позитивных = {baseline:.2f})')

    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title('Кривая Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Визуализация кривой Precision-Recall сохранена в '{output_path}'")


def visualize_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                n_bins: int = 10,
                                output_path: str = 'calibration_curve.png') -> None:
    """
    Визуализация кривой калибровки модели

    Args:
        y_true: Истинные целевые значения
        y_pred_proba: Предсказанные вероятности
        n_bins: Количество бинов для калибровочной кривой
        output_path: Путь для сохранения изображения
    """
    # Расчет кривой калибровки
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    # Построение кривой калибровки
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='LightGBM')

    # Построение идеальной кривой калибровки
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Идеально калиброванная')

    plt.xlabel('Средняя предсказанная вероятность')
    plt.ylabel('Доля положительных исходов')
    plt.title('Кривая калибровки')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Кривая калибровки сохранена в '{output_path}'")


def visualize_learning_curve(
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        output_path: str = 'learning_curve.png',
        random_state: int = 42) -> None:
    """
    Визуализация кривой обучения

    Args:
        estimator: Модель для обучения
        X_train: DataFrame с признаками обучающей выборки
        y_train: Серия целевых значений обучающей выборки
        cv: Количество фолдов для кросс-валидации
        output_path: Путь для сохранения изображения
        random_state: Seed для воспроизводимости результатов
    """
    # Проверка наличия обоих классов
    class_counts = y_train.value_counts()
    logger.info(f"Визуализация кривой обучения. Распределение классов: {class_counts.to_dict()}")

    if len(class_counts) < 2 or 1 not in class_counts or 0 not in class_counts:
        logger.error(f"ОШИБКА! В данных отсутствует один из классов: {class_counts.to_dict()}")
        logger.error("Невозможно построить кривую обучения с метрикой F1")

        # Создаем пустое изображение с сообщением об ошибке
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Ошибка: невозможно построить кривую обучения.\nВ данных отсутствует один из классов.',
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        return

    try:
        # Расчет кривой обучения с настройкой метрики
        # Используем accuracy вместо f1, если в данных мало примеров положительного класса
        if class_counts.get(1, 0) < 5:
            logger.warning(
                f"Мало примеров положительного класса ({class_counts.get(1, 0)}). Используем метрику accuracy.")
            scoring = 'accuracy'
        else:
            scoring = 'f1'

        logger.info(f"Расчет кривой обучения с метрикой: {scoring}")

        stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        # Проверка всех фолдов на представленность классов
        fold_issues = []
        for i, (train_idx, test_idx) in enumerate(stratified_kfold.split(X_train, y_train)):
            y_train_fold = y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]

            # Проверка наличия всех классов в обучающей и тестовой выборках
            if len(y_train_fold.unique()) < 2:
                fold_issues.append(f"Фолд {i+1}: в обучающей выборке отсутствует один из классов: {set(y_train_fold.unique())}")
            if len(y_test_fold.unique()) < 2:
                fold_issues.append(f"Фолд {i+1}: в тестовой выборке отсутствует один из классов: {set(y_test_fold.unique())}")

        # Если есть проблемы с представленностью классов в фолдах
        if fold_issues:
            for issue in fold_issues:
                logger.warning(issue)
            logger.info("Попытка использования альтернативной стратегии с принудительной стратификацией")

            # Используем более надежную стратегию для очень несбалансированных данных
            # Из библиотеки imblearn или просто используем accuracy метрику
            if scoring == 'f1':
                logger.info("Переключаемся на метрику accuracy из-за проблем со стратификацией")
                scoring = 'accuracy'

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X_train, y_train, cv=stratified_kfold, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring=scoring
        )

        # Расчет среднего и стандартного отклонения
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Построение кривой обучения
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Оценка на обучающей выборке')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Оценка на кросс-валидации')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.xlabel('Количество обучающих примеров')
        plt.ylabel(f'Метрика: {scoring}')
        plt.title('Кривая обучения')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Кривая обучения сохранена в '{output_path}'")
    except Exception as e:
        logger.error(f"Ошибка при создании кривой обучения: {e}")
        # Создаем пустое изображение с сообщением об ошибке
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Ошибка: {str(e)}',
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()


def visualize_lift(bin_metrics: pd.DataFrame, output_path: str = 'lift_charts.png') -> None:
    """
    Визуализация lift-диаграммы

    Args:
        bin_metrics: DataFrame с данными для lift-диаграммы
        output_path: Путь для сохранения изображения
    """
    plt.figure(figsize=(12, 10))

    # Создание подграфика для диаграммы Lift
    plt.subplot(2, 1, 1)
    plt.bar(bin_metrics['bin'], bin_metrics['lift'], color='skyblue')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.xlabel('Бин (отсортировано по предсказанной вероятности)')
    plt.ylabel('Lift')
    plt.title('Lift-диаграмма')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Создание подграфика для кумулятивного Lift
    plt.subplot(2, 1, 2)
    plt.plot(bin_metrics['bin'], bin_metrics['cumulative_lift'], marker='o', color='green')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.xlabel('Бин (отсортировано по предсказанной вероятности)')
    plt.ylabel('Кумулятивный Lift')
    plt.title('Диаграмма кумулятивного Lift')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Визуализации lift-диаграмм сохранены в '{output_path}'")
