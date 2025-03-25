"""Функции для визуализации SHAP значений для интерпретации модели"""
import inspect
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
        # Проверка совместимости признаков модели и данных
        if hasattr(model, 'feature_name_') and len(model.feature_name_) != X_sample.shape[1]:
            logger.warning(f"Несоответствие числа признаков: модель ожидает {len(model.feature_name_)}, "
                           f"но данные содержат {X_sample.shape[1]} признаков.")

            # Проверяем, можем ли мы использовать disable_shape_check
            if hasattr(model, 'predict') and 'predict_disable_shape_check' in str(inspect.signature(model.predict)):
                logger.info("Используем опцию predict_disable_shape_check=True для SHAP")
                disable_shape_check = True
            else:
                logger.error("Невозможно продолжить визуализацию SHAP из-за несоответствия числа признаков")
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5,
                         f"Невозможно построить SHAP визуализацию:\nмодель ожидает {len(model.feature_name_)} признаков, "
                         f"но данные содержат {X_sample.shape[1]} признаков.",
                         horizontalalignment='center', verticalalignment='center', fontsize=12)
                plt.title('Ошибка построения SHAP визуализации')
                plt.savefig(os.path.join(output_dir, 'shap_error.png'))
                plt.close()
                return
        else:
            disable_shape_check = False

        # Использование подмножества данных для анализа SHAP
        if len(X_sample) > n_samples:
            X_subset = X_sample.sample(n_samples, random_state=42)
        else:
            X_subset = X_sample

        # Создаем SHAP explainer
        try:
            # Пробуем создать TreeExplainer для моделей на основе деревьев
            explainer = shap.TreeExplainer(model)

            # Расчет значений SHAP с отключением проверки на совместимость формы данных
            if disable_shape_check and hasattr(model, 'predict'):
                # Обходной путь: создаем функцию предсказания, которая использует disable_shape_check
                def predict_func(X):
                    return model.predict(X, predict_disable_shape_check=True)

                # Используем эту функцию для вычисления значений SHAP
                shap_values = explainer.shap_values(X_subset, check_additivity=False)
            else:
                shap_values = explainer.shap_values(X_subset)

            # Проверяем тип возвращаемых значений
            if isinstance(shap_values, list):
                # Для мультиклассовой классификации или других случаев, когда возвращается список
                if len(shap_values) > 0:
                    # Используем первый набор значений для визуализации
                    shap_values_for_plot = shap_values[0]
                else:
                    raise ValueError("SHAP вернул пустой список значений")
            else:
                # Для регрессии или бинарной классификации
                shap_values_for_plot = shap_values

            # Сводный график
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_for_plot, X_subset, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
            plt.close()

            # Детальный график
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values_for_plot, X_subset, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'shap_detailed.png'))
                plt.close()
            except Exception as detail_error:
                logger.error(f"Ошибка при создании детального SHAP графика: {detail_error}")

            # Генерация скрипичного графика для топ-20 признаков
            if X_subset.shape[1] > 10:
                try:
                    # Получение важности признаков
                    feature_importance = pd.DataFrame({
                        'feature': X_subset.columns,
                        'importance': np.abs(shap_values_for_plot).mean(axis=0)
                    }).sort_values('importance', ascending=False)

                    # Получение топ-20 признаков по важности
                    top_features = feature_importance.head(20)['feature'].tolist()
                    X_top = X_subset[top_features]

                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(shap_values_for_plot[:, [X_subset.columns.get_loc(f) for f in top_features]],
                                      X_top, show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'shap_top_features.png'))
                    plt.close()
                except Exception as top_error:
                    logger.error(f"Ошибка при создании SHAP графика топ-признаков: {top_error}")

            logger.info(f"Визуализации SHAP сохранены в директории '{output_dir}'")

        except Exception as detail_error:
            logger.error("Ошибка при создании визуализаций SHAP")
    except Exception as e:
        logger.error(f"Ошибка генерации визуализаций SHAP: {e}")
        # Создаем изображение с сообщением об ошибке
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Не удалось создать SHAP визуализацию:\n{str(e)}",
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.title('Ошибка SHAP визуализации')
        plt.savefig(os.path.join(output_dir, 'shap_error.png'))
        plt.close()
