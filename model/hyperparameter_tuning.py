"""Функции для оптимизации гиперпараметров модели LightGBM"""

import pandas as pd
import logging
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from typing import Dict, Callable

logger = logging.getLogger(__name__)


def objective_factory(X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      random_state: int = 42) -> Callable:
    """
    Создает функцию цели для оптимизации Optuna

    Args:
        X_train: DataFrame с признаками обучающей выборки
        y_train: Серия целевых значений обучающей выборки
        X_val: DataFrame с признаками валидационной выборки
        y_val: Серия целевых значений валидационной выборки
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

        # Создание объектов датасета
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Обучение модели
        callbacks = [
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=callbacks
        )

        # Получение предсказаний на валидационной выборке
        y_pred_val = model.predict(X_val)

        # Расчет F1-score (наша метрика оптимизации)
        y_pred_binary = (y_pred_val > 0.5).astype(int)
        f1 = f1_score(y_val, y_pred_binary)

        return f1

    return objective


def optimize_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             n_trials: int = 50, random_state: int = 42) -> Dict:
    """
    Оптимизация гиперпараметров с использованием Optuna

    Args:
        X_train: DataFrame с признаками обучающей выборки
        y_train: Серия целевых значений обучающей выборки
        X_val: DataFrame с признаками валидационной выборки
        y_val: Серия целевых значений валидационной выборки
        n_trials: Количество испытаний для оптимизации
        random_state: Seed для генератора случайных чисел

    Returns:
        Dict: Лучшие гиперпараметры
    """
    logger.info(f"Оптимизация гиперпараметров с {n_trials} испытаниями")

    # Создание функции цели для Optuna
    objective = objective_factory(X_train, y_train, X_val, y_val, random_state)

    # Создание и запуск исследования с параллельной обработкой
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    # Получение лучших параметров
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'binary_logloss'
    best_params['boosting_type'] = 'gbdt'
    best_params['random_state'] = random_state
    # GPU настройки
    best_params['device'] = 'gpu'
    best_params['gpu_platform_id'] = 0
    best_params['gpu_device_id'] = 0
    best_params['gpu_use_dp'] = False
    best_params['max_bin'] = 63

    logger.info(f"Лучшие гиперпараметры: {best_params}")
    logger.info(f"Лучший F1 score: {study.best_value:.4f}")

    return best_params
