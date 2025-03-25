"""Функции для оптимизации гиперпараметров модели LightGBM"""

import numpy as np
import pandas as pd
import logging
import lightgbm as lgb
from sklearn.metrics import f1_score
from typing import Callable


logger = logging.getLogger(__name__)


def objective_factory(
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series,
        kfold,
        random_state: int = 42) -> Callable:
    """
    Создает функцию цели для оптимизации Optuna с использованием кросс-валидации

    Args:
        X_train_val: DataFrame с признаками обучающей+валидационной выборки
        y_train_val: Серия целевых значений обучающей+валидационной выборки
        kfold: Объект кросс-валидации
        random_state: Seed для генератора случайных чисел

    Returns:
        Callable: Функция цели для Optuna
    """

    def objective(trial):
        """
        Функция цели для оптимизации Optuna

        Args:
            trial: Объект trial Optuna

        Returns:
            float: Значение метрики (F1 score)
        """
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'random_state': random_state,
            # Настройки GPU
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'gpu_use_dp': False,
            'max_bin': 63,
            'verbose': -1
        }

        # Инициализация массива для хранения F1-scores по фолдам
        f1_scores = []

        # Кросс-валидация
        for train_idx, val_idx in kfold.split(X_train_val, y_train_val):
            # Разделение данных для текущего фолда
            X_train_fold, X_val_fold = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_train_fold, y_val_fold = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

            # Создание объектов датасета
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

            # Обучение модели
            callbacks = [
                lgb.early_stopping(50)
            ]

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=callbacks
            )

            # Получение предсказаний на валидационной выборке
            y_pred_val = model.predict(X_val_fold)

            # Расчет F1-score
            y_pred_binary = (y_pred_val > 0.5).astype(int)
            f1 = f1_score(y_val_fold, y_pred_binary)
            f1_scores.append(f1)

        # Возвращаем среднее значение F1-score по всем фолдам
        return np.mean(f1_scores)

    return objective
