"""Функции для визуализации метрик производительности модели"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, StratifiedKFold
from typing import Any, Optional, Dict

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
        random_state: int = 42,
        final_params: Optional[Dict] = None) -> None:
    """
    Визуализация кривой обучения с использованием оптимальных параметров модели

    Args:
        estimator: Базовая модель для создания клона с оптимальными параметрами
        X_train: DataFrame с признаками обучающей выборки
        y_train: Серия целевых значений обучающей выборки
        cv: Количество фолдов для кросс-валидации
        output_path: Путь для сохранения изображения
        random_state: Seed для воспроизводимости результатов
        final_params: Словарь с оптимальными параметрами модели
    """
    logger.info("Визуализация кривой обучения с F1-метрикой")

    # Проверка наличия обоих классов
    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or min(class_counts.values) < 3:
        logger.error(f"Недостаточно данных для F1-метрики: {class_counts.to_dict()}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Ошибка: недостаточно данных для построения F1-кривой обучения',
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        return

    try:
        # Создание клона модели с оптимальными параметрами
        model_params = final_params.copy() if final_params else {}
        # Удаляем служебные параметры
        for param in ['random_state', 'verbose']:
            if param in model_params:
                del model_params[param]

        # Добавляем random_state для воспроизводимости
        model_params['random_state'] = random_state
        for key, value in model_params.items():
            setattr(estimator, key, value)

        # Определение стратегии кросс-валидации
        stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        # Расчет кривой обучения с F1-метрикой
        train_sizes = np.linspace(0.1, 1.0, 15)  # Увеличиваем число точек для более гладкой кривой
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X_train, y_train, cv=stratified_kfold, n_jobs=-1,
            train_sizes=train_sizes,
            scoring='f1',
            random_state=random_state
        )

        # Расчет среднего и стандартного отклонения
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Установка стиля
        with plt.style.context('seaborn-v0_8-whitegrid'):
            fig, ax = plt.subplots(figsize=(12, 8))

            # Главные кривые обучения
            ax.plot(train_sizes_abs, train_mean, 'o-', color='#1f77b4', linewidth=2.5,
                    markersize=8, label='Обучающая выборка')
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                            alpha=0.2, color='#1f77b4')

            ax.plot(train_sizes_abs, test_mean, 's-', color='#ff7f0e', linewidth=2.5,
                    markersize=8, label='Кросс-валидация')
            ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                            alpha=0.2, color='#ff7f0e')

            # Базовая линия (случайная модель для бинарной классификации)
            positive_class_ratio = y_train.mean()
            baseline_f1 = 2 * positive_class_ratio / (1 + positive_class_ratio)
            ax.axhline(y=baseline_f1, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8,
                       label=f'Базовая линия (F1={baseline_f1:.3f})')

            # Разрыв между кривыми (overfitting gap)
            for i in range(len(train_sizes_abs)):
                if i % 3 == 0 or i == len(train_sizes_abs) - 1:  # Рисуем не все линии для ясности
                    gap = train_mean[i] - test_mean[i]
                    ax.plot([train_sizes_abs[i], train_sizes_abs[i]],
                            [test_mean[i], train_mean[i]], 'k--', alpha=0.3, linewidth=1)
                    if gap > 0.1:  # Показываем метку только для существенных разрывов
                        ax.text(train_sizes_abs[i], test_mean[i] + gap / 2, f'{gap:.2f}',
                                horizontalalignment='left', fontsize=8)

            # Добавляем обозначение последней точки (максимальное количество примеров)
            final_gap = train_mean[-1] - test_mean[-1]
            ax.scatter(train_sizes_abs[-1], train_mean[-1], s=100, c='#1f77b4', zorder=5,
                       edgecolor='black', linewidth=1.5)
            ax.scatter(train_sizes_abs[-1], test_mean[-1], s=100, c='#ff7f0e', zorder=5,
                       edgecolor='black', linewidth=1.5)

            # Аннотации для последних точек
            ax.annotate(f'F1={train_mean[-1]:.3f}±{train_std[-1]:.3f}',
                        xy=(train_sizes_abs[-1], train_mean[-1]),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

            ax.annotate(f'F1={test_mean[-1]:.3f}±{test_std[-1]:.3f}',
                        xy=(train_sizes_abs[-1], test_mean[-1]),
                        xytext=(10, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

            # Оформление графика
            ax.set_xlabel('Количество обучающих примеров', fontsize=14)
            ax.set_ylabel('F1-метрика', fontsize=14)
            ax.set_title('Кривая обучения', fontsize=16)

            # Добавляем информацию о модели в виде текстового блока
            param_text = []
            if final_params:
                important_params = ['num_leaves', 'learning_rate', 'max_depth', 'lambda_l1', 'lambda_l2']
                for param in important_params:
                    if param in final_params:
                        param_text.append(f"{param}: {final_params[param]}")

            param_info = '\n'.join(param_text[:5])  # Ограничиваем количество параметров для читаемости

            # Информация о данных и модели
            plt.figtext(0.02, 0.02,
                        f"Данные: {len(X_train)} примеров, {X_train.shape[1]} признаков\n"
                        f"Баланс классов: {class_counts.get(1, 0)}/{class_counts.get(0, 0)}\n"
                        f"Параметры модели:\n{param_info}",
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            # Легенда и сетка
            ax.legend(loc='lower right', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Улучшенные отметки осей
            ax.tick_params(axis='both', which='major', labelsize=12)

            # Добавляем информацию об overfitting
            final_gap_text = (f"Разрыв Train-Val: {final_gap:.3f} "
                              f"({'Высокий' if final_gap > 0.25 else 'Средний' if final_gap > 0.1 else 'Низкий'} уровень переобучения)")
            plt.figtext(0.5, 0.01, final_gap_text, fontsize=12, ha='center',
                        bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.5'))

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Кривая обучения сохранена в '{output_path}'")
        logger.info(f"Финальный F1 на обучающей выборке: {train_mean[-1]:.4f}±{train_std[-1]:.4f}")
        logger.info(f"Финальный F1 на валидации: {test_mean[-1]:.4f}±{test_std[-1]:.4f}")
        logger.info(f"Разрыв между обучающим и валидационным F1: {final_gap:.4f}")

    except Exception as e:
        logger.error(f"Ошибка при создании кривой обучения: {e}")
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
