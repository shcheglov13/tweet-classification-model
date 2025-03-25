"""
Основной класс модели для классификации твитов по пригодности для токенизации
"""

import os
import numpy as np
import pandas as pd
import logging
import lightgbm as lgb
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .preprocessing import preprocess_data
from .feature_selection import analyze_correlations, group_features, select_features_from_model
from .model_training import train_model, find_optimal_threshold, predict, split_data
from .hyperparameter_tuning import optimize_hyperparameters
from .evaluation import evaluate_model, compute_lift

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_io import save_model, load_model

logger = logging.getLogger(__name__)


class TokenizatorModel:
    """
    Класс для классификации твитов по пригодности для токенизации

    Attributes:
        random_state (int): Seed для генератора случайных чисел
        model: Обученная модель LightGBM
        feature_importance (pd.DataFrame): DataFrame с важностью признаков
        best_threshold (float): Оптимальный порог для классификации
        feature_names (List[str]): Имена признаков
        scaler (StandardScaler): Масштабировщик признаков
        selected_features (List[str]): Выбранные признаки
        feature_groups (Dict[str, List[str]]): Группы признаков
        model_metrics (Dict): Метрики модели
        fold_results (Dict): Результаты по фолдам кросс-валидации
    """

    def __init__(self, random_state: int = 42):
        """
        Инициализация модели

        Args:
            random_state: Seed для генератора случайных чисел
        """
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.best_threshold = 0.5
        self.feature_names = None
        self.scaler = None
        self.selected_features = None
        self.feature_groups = None
        self.model_metrics = {}
        self.fold_results = {}

    def preprocess_data(
            self, X: pd.DataFrame,
            y: Optional[pd.Series] = None,
            threshold: float = 100) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """
        Предобработка данных путем масштабирования и опционально бинаризации

        Args:
            X: DataFrame с признаками
            y: Серия целевых значений (опционально)
            threshold: Порог для бинаризации

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
                Масштабированные признаки и опционально бинаризованная цель
        """
        logger.info(f"Предобработка данных с порогом {threshold}")

        # Вызов функции предобработки
        X_scaled, y_binary, self.scaler = preprocess_data(X, y, threshold, self.scaler)

        # Возвращаем в зависимости от наличия y
        if y is not None:
            return X_scaled, y_binary
        return X_scaled

    def analyze_feature_correlations(self, X: pd.DataFrame, threshold: float = 0.9) -> Tuple[pd.DataFrame, List[str]]:
        """
        Анализ и удаление сильно коррелирующих признаков

        Args:
            X: DataFrame с признаками
            threshold: Порог корреляции для удаления

        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame с выбранными признаками и список удаленных признаков
        """
        logger.info(f"Анализ корреляций признаков с порогом {threshold}")

        # Вызов функции анализа корреляций
        X_reduced, to_drop = analyze_correlations(X, threshold)

        return X_reduced, to_drop

    def select_features(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            kfold,
            k: int = 100) -> Tuple[pd.DataFrame, List[str]]:
        """
        Выбор топ-k признаков с использованием кросс-валидации

        Args:
            X_train_val: DataFrame с признаками обучающей+валидационной выборки
            y_train_val: Серия целевых значений обучающей+валидационной выборки
            kfold: Объект кросс-валидации
            k: Количество признаков для выбора

        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame с выбранными признаками и список выбранных признаков
        """
        logger.info(f"Выбор топ-{k} признаков с использованием кросс-валидации")

        X_selected, selected_features = select_features_from_model(
            X_train_val, y_train_val, kfold, random_state=self.random_state, k=k)

        # Сохранение выбранных признаков
        self.selected_features = selected_features

        return X_selected, selected_features

    def group_features(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Группировка признаков по типу

        Args:
            X: DataFrame с признаками

        Returns:
            Dict[str, List[str]]: Словарь с группами признаков
        """
        logger.info("Группировка признаков по типу")

        # Вызов функции группировки признаков
        self.feature_groups = group_features(X)

        return self.feature_groups

    def incremental_feature_evaluation(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            kfold) -> Dict:
        """
        Инкрементальная оценка групп признаков для измерения их вклада с использованием кросс-валидации

        Args:
            X_train_val: DataFrame с признаками обучающей+валидационной выборки
            y_train_val: Серия целевых значений обучающей+валидационной выборки
            kfold: Объект кросс-валидации

        Returns:
            Dict: Словарь с результатами инкрементальной оценки
        """
        logger.info("Инкрементальная оценка групп признаков с кросс-валидацией")

        if not self.feature_groups:
            logger.error("Группы признаков не определены. Сначала вызовите group_features.")
            return None

        # Определение базовой модели для оценки
        base_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            random_state=self.random_state,
            n_estimators=100,
            # Настройки GPU
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            gpu_use_dp=False,
            max_bin=63
        )

        # Оценка вклада каждой группы
        results = {}
        current_columns = []

        # Обработка групп признаков в определенном порядке для инкрементальной оценки
        ordered_groups = [
            'text_metrics', 'bertweet', 'informal_slang',
            'structural', 'temporal', 'clip', 'emotional', 'other'
        ]

        # Инкрементальная оценка каждой группы
        for group in ordered_groups:
            if group in self.feature_groups:
                current_columns.extend(self.feature_groups[group])

                # Создание текущего набора признаков
                X_current = X_train_val[current_columns]

                # Инициализация массивов для хранения метрик по фолдам
                fold_metrics = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'roc_auc': []
                }

                # Обучение и оценка на каждом фолде
                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
                    # Разделение данных для текущего фолда
                    X_train_fold = X_current.iloc[train_idx]
                    y_train_fold = y_train_val.iloc[train_idx]
                    X_val_fold = X_current.iloc[val_idx]
                    y_val_fold = y_train_val.iloc[val_idx]

                    # Обучение модели
                    model = lgb.LGBMClassifier(**base_model.get_params())
                    model.fit(X_train_fold, y_train_fold)

                    # Получение предсказаний
                    y_pred = model.predict(X_val_fold)
                    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

                    # Расчет метрик для текущего фолда
                    fold_metrics['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                    fold_metrics['precision'].append(precision_score(y_val_fold, y_pred))
                    fold_metrics['recall'].append(recall_score(y_val_fold, y_pred))
                    fold_metrics['f1'].append(f1_score(y_val_fold, y_pred))
                    fold_metrics['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))

                # Расчет средних метрик по всем фолдам
                results[group] = {
                    'accuracy': np.mean(fold_metrics['accuracy']),
                    'precision': np.mean(fold_metrics['precision']),
                    'recall': np.mean(fold_metrics['recall']),
                    'f1': np.mean(fold_metrics['f1']),
                    'roc_auc': np.mean(fold_metrics['roc_auc']),
                    'features_count': len(current_columns)
                }

                logger.info(f"Добавлена группа '{group}': F1 = {results[group]['f1']:.4f}, "
                            f"Всего признаков = {results[group]['features_count']}")

        return results

    def split_data(
            self, X: pd.DataFrame,
            y: pd.Series,
            n_splits: int = 5,
            test_size: float = 0.2) -> Tuple:
        """
        Разделение данных на тестовую выборку и кроссвалидационные разбиения
        обучающей+валидационной выборки

        Args:
            X: DataFrame с признаками
            y: Серия целевых значений
            n_splits: Количество фолдов для кросс-валидации
            test_size: Размер тестовой выборки (0-1)

        Returns:
            Tuple: X_train_val, X_test, y_train_val, y_test, kfold (StratifiedKFold object)
        """

        logger.info(f"Разделение данных на тестовую выборку и {n_splits}-фолдовую кросс-валидацию")

        return split_data(X, y, n_splits, test_size, self.random_state)

    def train(self, X_train_val: pd.DataFrame, y_train_val: pd.Series,
              kfold, params: Optional[Dict] = None) -> Dict:
        """
        Обучение модели LightGBM с использованием кросс-валидации

        Args:
            X_train_val: DataFrame с признаками обучающей и валидационной выборки
            y_train_val: Серия целевых значений обучающей и валидационной выборки
            kfold: Объект кросс-валидации
            params: Параметры модели LightGBM

        Returns:
            Dict: Результаты кросс-валидации
        """
        logger.info("Обучение модели LightGBM с кросс-валидацией")

        # Сохранение имен признаков
        self.feature_names = X_train_val.columns.tolist()

        self.model, self.feature_importance, self.fold_results = train_model(
            X_train_val, y_train_val, kfold, params, self.feature_names)

        logger.info(f"Модель обучена с {len(self.feature_names)} признаками")

        return self.fold_results

    def optimize_hyperparameters(self, X_train_val: pd.DataFrame, y_train_val: pd.Series,
                                            kfold, n_trials: int = 50) -> Dict:
        """
        Оптимизация гиперпараметров с использованием Optuna и кросс-валидации

        Args:
            X_train_val: DataFrame с признаками обучающей+валидационной выборки
            y_train_val: Серия целевых значений обучающей+валидационной выборки
            kfold: Объект кросс-валидации
            n_trials: Количество испытаний для оптимизации

        Returns:
            Dict: Лучшие гиперпараметры
        """
        logger.info(f"Оптимизация гиперпараметров с {n_trials} испытаниями и кросс-валидацией")

        best_params = optimize_hyperparameters(
            X_train_val, y_train_val, kfold, n_trials, self.random_state)

        # Обучение модели с лучшими параметрами
        return self.train(X_train_val, y_train_val, kfold, best_params)

    def find_optimal_threshold(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            kfold) -> float:
        """
        Поиск оптимального порога классификации с использованием кросс-валидации

        Args:
            X_train_val: DataFrame с признаками обучающей+валидационной выборки
            y_train_val: Серия целевых значений обучающей+валидационной выборки
            kfold: Объект кросс-валидации

        Returns:
            float: Оптимальный порог
        """
        logger.info("Поиск оптимального порога классификации с кросс-валидацией")

        # Проверка наличия модели
        if not self.model:
            logger.error("Модель еще не обучена")
            return self.best_threshold

        self.best_threshold = find_optimal_threshold(self.model, X_train_val, y_train_val, kfold)

        return self.best_threshold

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: Optional[float] = None) -> Dict:
        """
        Оценка модели на заданных данных

        Args:
            X: DataFrame с признаками
            y: Серия целевых значений
            threshold: Порог классификации

        Returns:
            Dict: Словарь с метриками
        """
        if not self.model:
            logger.error("Модель еще не обучена")
            return None

        if threshold is None:
            threshold = self.best_threshold

        logger.info(f"Оценка модели с порогом {threshold}")

        # Вызов функции оценки модели
        metrics = evaluate_model(self.model, X, y, threshold)

        # Сохранение метрик для дальнейшего использования
        self.model_metrics = metrics

        return metrics

    def compute_lift(self, X: pd.DataFrame, y: pd.Series, bins: int = 10) -> pd.DataFrame:
        """
        Расчет данных для lift-диаграммы

        Args:
            X: DataFrame с признаками
            y: Серия целевых значений
            bins: Количество бинов для диаграммы

        Returns:
            pd.DataFrame: DataFrame с данными для lift-диаграммы
        """
        logger.info(f"Расчет lift-диаграммы с {bins} бинами")

        # Проверка наличия модели
        if not self.model:
            logger.error("Модель еще не обучена")
            return None

        # Вызов функции расчета лифта
        return compute_lift(self.model, X, y, bins)

    def save_model(self, filepath: str = 'tokenizator_model.txt') -> bool:
        """
        Сохранение модели в файл

        Args:
            filepath: Путь для сохранения модели

        Returns:
            bool: Статус сохранения
        """
        logger.info(f"Сохранение модели в {filepath}")

        # Проверка наличия модели
        if not self.model:
            logger.error("Модель еще не обучена")
            return False

        # Вызов функции сохранения модели
        return save_model(
            self.model,
            filepath,
            self.best_threshold,
            self.feature_names,
            self.selected_features,
            self.feature_groups,
            self.model_metrics,
            self.scaler,
            self.feature_importance
        )

    def load_model(self, filepath: str = 'tokenizator_model.txt') -> bool:
        """
        Загрузка модели из файла

        Args:
            filepath: Путь к файлу модели

        Returns:
            bool: Статус загрузки
        """
        logger.info(f"Загрузка модели из {filepath}")

        try:
            # Вызов функции загрузки модели
            self.model, model_info, self.scaler, self.feature_importance = load_model(filepath)

            if self.model:
                # Загрузка дополнительной информации
                self.best_threshold = model_info.get('best_threshold', 0.5)
                self.feature_names = model_info.get('feature_names', None)
                self.selected_features = model_info.get('selected_features', None)
                self.feature_groups = model_info.get('feature_groups', None)
                self.model_metrics = model_info.get('model_metrics', {})
                return True

            return False

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание на новых данных

        Args:
            X: DataFrame с признаками
            threshold: Порог классификации

        Returns:
            Tuple[np.ndarray, np.ndarray]: Бинарные предсказания и вероятности
        """
        if not self.model:
            logger.error("Модель еще не обучена")
            return None, None

        if threshold is None:
            threshold = self.best_threshold

        logger.info(f"Выполнение предсказаний с порогом {threshold}")

        # Масштабирование признаков
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X

        # Использование только выбранных признаков, если доступны
        if self.selected_features is not None and set(self.selected_features).issubset(set(X_scaled.columns)):
            X_pred = X_scaled[self.selected_features]
        else:
            X_pred = X_scaled

        # Выполнение предсказаний
        return predict(self.model, X_pred, threshold)
