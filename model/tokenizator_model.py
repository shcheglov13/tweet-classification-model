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
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

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
        self.sorted_group_order = None
        self.incremental_results = None
        self.best_params = None
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

    def select_features_with_optimal_groups(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            kfold,
            optimal_group_order: List[str],
            k: int = 100) -> Tuple[pd.DataFrame, List[str]]:
        """
        Выбор признаков с использованием оптимизированного порядка групп

        Args:
            X_train_val: DataFrame с признаками
            y_train_val: Серия целевых значений
            kfold: Объект кросс-валидации
            optimal_group_order: Оптимизированный порядок групп признаков
            k: Максимальное количество признаков

        Returns:
            Tuple[pd.DataFrame, List[str]]: Выбранные признаки и их список
        """
        logger.info(f"Выбор признаков с оптимизированным порядком групп признаков, max_k={k}")

        if not self.feature_groups:
            self.feature_groups = self.group_features(X_train_val)

        # Поочередно добавляем группы в приоритетном порядке
        all_selected_features = []

        for group_name in optimal_group_order:
            if group_name not in self.feature_groups:
                continue

            # Добавляем признаки из текущей группы
            group_features = self.feature_groups[group_name]
            all_selected_features.extend(group_features)

            # Если превысили лимит, обрезаем список
            if len(all_selected_features) > k:
                # Выбор наиболее важных признаков через кросс-валидацию
                _, important_features = select_features_from_model(
                    X_train_val[all_selected_features],
                    y_train_val,
                    kfold,
                    random_state=self.random_state,
                    k=k
                )
                all_selected_features = important_features

        self.selected_features = all_selected_features
        X_selected = X_train_val[all_selected_features]

        logger.info(f"Выбрано {len(all_selected_features)} признаков")

        return X_selected, all_selected_features

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
        Инкрементальная оценка групп признаков для измерения их вклада
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
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            gpu_use_dp=False,
            max_bin=63,
            verbose=-1,
        )

        # Оценка вклада каждой группы
        results = {}
        current_columns = []
        group_contributions = {}

        # Обработка групп признаков в определенном порядке
        ordered_groups = [
            'text_metrics', 'bertweet', 'informal_slang',
            'structural', 'temporal', 'clip', 'emotional', 'other'
        ]

        # Фильтрация только существующих групп
        ordered_groups = [g for g in ordered_groups if g in self.feature_groups]

        # Инкрементальная оценка каждой группы
        baseline_f1 = 0

        for group in ordered_groups:
            # Добавление признаков текущей группы
            current_columns.extend(self.feature_groups[group])

            # Создание текущего набора признаков
            X_current = X_train_val[current_columns]

            # Инициализация массивов для хранения метрик по фолдам
            fold_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            }

            # Обучение и оценка на каждом фолде
            for train_idx, val_idx in kfold.split(X_train_val, y_train_val):
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

            # Расчет средних метрик по всем фолдам
            current_f1 = np.mean(fold_metrics['f1'])

            # Расчет вклада этой группы (прирост F1)
            contribution = current_f1 - baseline_f1
            group_contributions[group] = contribution
            baseline_f1 = current_f1

            results[group] = {
                'accuracy': np.mean(fold_metrics['accuracy']),
                'precision': np.mean(fold_metrics['precision']),
                'recall': np.mean(fold_metrics['recall']),
                'f1': current_f1,
                'features_count': len(current_columns),
                'contribution': contribution
            }

            logger.info(f"Добавлена группа '{group}': F1 = {current_f1:.4f}, "
                        f"Вклад = {contribution:.4f}, "
                        f"Всего признаков = {len(current_columns)}")

        # Сортировка групп по вкладу
        sorted_groups = sorted(group_contributions.items(), key=lambda x: x[1], reverse=True)
        logger.info("Группы признаков по вкладу в метрику F1:")
        for group, contrib in sorted_groups:
            logger.info(f"  {group}: {contrib:.4f}")

        # Сохраняем результаты и отсортированный порядок групп
        self.incremental_results = results
        self.sorted_group_order = [g for g, _ in sorted_groups]

        return results

    def apply_incremental_evaluation_results(
            self,
            X_train_val: pd.DataFrame,
            min_contribution: float = 0.001) -> pd.DataFrame:
        """
        Применение результатов инкрементальной оценки для выбора признаков

        Args:
            X_train_val: DataFrame с признаками
            min_contribution: Минимальный вклад группы для включения

        Returns:
            pd.DataFrame: DataFrame с выбранными признаками
        """
        if not hasattr(self, 'incremental_results') or not self.incremental_results:
            logger.error(
                "Отсутствуют результаты инкрементальной оценки. Сначала выполните incremental_feature_evaluation.")
            return X_train_val

        if not hasattr(self, 'sorted_group_order') or not self.sorted_group_order:
            logger.error("Отсутствует отсортированный порядок групп. Сначала выполните incremental_feature_evaluation.")
            return X_train_val

        # Выбор групп с вкладом выше порога
        significant_groups = []
        for group in self.sorted_group_order:
            contribution = self.incremental_results[group]['contribution']
            if contribution >= min_contribution:
                significant_groups.append(group)
                logger.info(f"Группа '{group}' включена (вклад: {contribution:.4f})")
            else:
                logger.info(f"Группа '{group}' исключена (вклад: {contribution:.4f} < {min_contribution})")

        # Сбор признаков из выбранных групп
        selected_features = []
        for group in significant_groups:
            selected_features.extend(self.feature_groups[group])

        logger.info(f"На основе инкрементальной оценки выбрано {len(selected_features)} признаков")
        self.selected_features = selected_features

        # Обновляем feature_groups, удаляя исключенные группы
        self.feature_groups = {group: self.feature_groups[group] for group in significant_groups}

        return X_train_val[selected_features]

    def split_data(
            self,
            X: pd.DataFrame,
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

    def train(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            kfold,
            params: Optional[Dict] = None,
            threshold: float = 0.5) -> Dict:
        """
        Обучение модели LightGBM с использованием кросс-валидации

        Args:
            :param X_train_val: DataFrame с признаками обучающей и валидационной выборки
            :param y_train_val: Серия целевых значений обучающей и валидационной выборки
            :param kfold: Объект кросс-валидации
            :param params: Параметры модели LightGBM
            :param threshold:

        Returns:
            Dict: Результаты кросс-валидации
        """
        logger.info(f"Обучение модели LightGBM с кросс-валидацией и порогом {threshold}")

        # Сохранение имен признаков
        self.feature_names = X_train_val.columns.tolist()

        # Обучение модели с заданным порогом
        self.model, self.feature_importance, self.fold_results = train_model(
            X_train_val, y_train_val, kfold, params, self.feature_names, threshold)

        logger.info(f"Модель обучена с {len(self.feature_names)} признаками")

        return self.fold_results

    def train_final_model(self, X_all: pd.DataFrame, y_all: pd.Series, params: Optional[Dict] = None) -> None:
        """
        Переобучение финальной модели на всех данных (тренировочных, валидационных, тестовых)

        Args:
            X_all: DataFrame со всеми признаками
            y_all: Серия всех целевых значений
            params: Параметры модели LightGBM
        """
        logger.info("Переобучение финальной модели на всех данных")

        # Использование оптимальных параметров, если они были найдены
        if params is None:
            if hasattr(self, 'best_params') and self.best_params:
                params = self.best_params
            else:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'lambda_l1': 0.1,
                    'lambda_l2': 0.1,
                    'min_child_samples': 20,
                    'max_depth': 8,
                    'random_state': self.random_state,
                    'verbose': -1
                }

        # Сохранение имен признаков
        self.feature_names = X_all.columns.tolist()

        # Создание датасета для обучения
        train_data = lgb.Dataset(X_all, label=y_all, feature_name=self.feature_names)

        # Обучение модели без валидационного набора
        final_model = lgb.train(
            params,
            train_data,
            num_boost_round=1000
        )

        # Сохранение финальной модели
        self.model = final_model

        # Расчет важности признаков
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        logger.info(f"Финальная модель обучена на {len(X_all)} примерах с {len(self.feature_names)} признаками")

    def optimize_hyperparameters(
            self,
            X_train_val:
            pd.DataFrame,
            y_train_val:
            pd.Series,
            kfold,
            n_trials: int = 50) -> Dict:
        """
        Оптимизация гиперпараметров с использованием Optuna и кросс-валидации
        """
        logger.info(f"Оптимизация гиперпараметров с {n_trials} испытаниями и кросс-валидацией")

        best_params = optimize_hyperparameters(
            X_train_val, y_train_val, kfold, n_trials, self.random_state)

        # Сохраняем оптимальные гиперпараметры как атрибут класса
        self.best_params = best_params

        return best_params

    def find_optimal_threshold(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            kfold,
            params: Optional[Dict] = None) -> float:
        """
        Поиск оптимального порога классификации для максимизации F1

        Args:
            X_train_val: DataFrame с признаками обучающей+валидационной выборки
            y_train_val: Серия целевых значений обучающей+валидационной выборки
            kfold: Объект кросс-валидации
            params: Параметры

        Returns:
            float: Оптимальный порог
        """
        logger.info("Поиск оптимального порога классификации с кросс-валидацией")

        # Использование оптимальных параметров, если они были найдены
        if params is None:
            if hasattr(self, 'best_params') and self.best_params:
                params = self.best_params
            else:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'random_state': self.random_state,
                    'verbose': -1
                }

        # Поиск оптимального порога с обучением на каждом фолде
        self.best_threshold = find_optimal_threshold(params, X_train_val, y_train_val, kfold)

        return self.best_threshold

    def analyze_fold_stability(self) -> Dict:
        """
        Анализ стабильности результатов между фолдами
        """
        if not self.fold_results:
            logger.error("Отсутствуют результаты по фолдам. Сначала обучите модель.")
            return {}

        return self.analyze_fold_stability_results(self.fold_results)

    def analyze_fold_stability_results(self, fold_results: Dict) -> Dict:
        """
        Анализ стабильности результатов между фолдами
        """
        logger.info("Анализ стабильности результатов между фолдами")

        # Извлечение метрик из каждого фолда
        fold_metrics = {
            'f1': [],
            'precision': [],
            'recall': []
        }

        for fold_idx, fold_data in fold_results.items():
            fold_metrics['f1'].append(fold_data['metrics']['f1'])
            fold_metrics['precision'].append(fold_data['metrics']['precision'])
            fold_metrics['recall'].append(fold_data['metrics']['recall'])

        # Расчет статистик стабильности
        stability_metrics = {
            'f1_mean': np.mean(fold_metrics['f1']),
            'f1_std': np.std(fold_metrics['f1']),
            'f1_cv': np.std(fold_metrics['f1']) / np.mean(fold_metrics['f1']),
            'precision_mean': np.mean(fold_metrics['precision']),
            'precision_std': np.std(fold_metrics['precision']),
            'precision_cv': np.std(fold_metrics['precision']) / np.mean(fold_metrics['precision']),
            'recall_mean': np.mean(fold_metrics['recall']),
            'recall_std': np.std(fold_metrics['recall']),
            'recall_cv': np.std(fold_metrics['recall']) / np.mean(fold_metrics['recall'])
        }

        logger.info(f"F1 score: {stability_metrics['f1_mean']:.4f} ± {stability_metrics['f1_std']:.4f} "
                    f"(CV = {stability_metrics['f1_cv']:.4f})")
        logger.info(f"Precision: {stability_metrics['precision_mean']:.4f} ± {stability_metrics['precision_std']:.4f} "
                    f"(CV = {stability_metrics['precision_cv']:.4f})")
        logger.info(f"Recall: {stability_metrics['recall_mean']:.4f} ± {stability_metrics['recall_std']:.4f} "
                    f"(CV = {stability_metrics['recall_cv']:.4f})")

        return stability_metrics

    def optimize_feature_groups_order_1(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            feature_groups: Dict[str, List[str]],
            kfold) -> List[str]:
        """
        Оптимизация порядка добавления групп признаков с использованием жадного алгоритма
        """
        logger.info("Оптимизация порядка добавления групп признаков")

        # Базовая модель для оценки
        base_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'random_state': 42,
            'verbose': -1
        }

        # Все доступные группы
        available_groups = list(feature_groups.keys())
        selected_groups = []
        current_features = []
        best_score = 0

        # Жадный алгоритм: последовательно добавляем группу, дающую наибольший прирост
        while available_groups:
            best_group = None
            best_group_score = best_score

            for group in available_groups:
                # Временно добавляем группу
                temp_features = current_features.copy()
                temp_features.extend(feature_groups[group])

                # Если нет признаков, пропускаем
                if not temp_features:
                    continue

                # Оценка на кросс-валидации
                fold_scores = []
                for train_idx, val_idx in kfold.split(X_train_val, y_train_val):
                    X_train_fold = X_train_val.iloc[train_idx][temp_features]
                    y_train_fold = y_train_val.iloc[train_idx]
                    X_val_fold = X_train_val.iloc[val_idx][temp_features]
                    y_val_fold = y_train_val.iloc[val_idx]

                    # Обучение модели
                    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

                    model = lgb.train(
                        base_params,
                        train_data,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(10)],
                        num_boost_round=100
                    )

                    # Оценка
                    y_pred = model.predict(X_val_fold)
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    f1 = f1_score(y_val_fold, y_pred_binary)
                    fold_scores.append(f1)

                # Среднее значение F1 по фолдам
                group_score = np.mean(fold_scores)

                # Обновляем лучшую группу, если нашли
                if group_score > best_group_score:
                    best_group = group
                    best_group_score = group_score

            # Если нашли группу, улучшающую результат, добавляем её
            if best_group:
                selected_groups.append(best_group)
                current_features.extend(feature_groups[best_group])
                available_groups.remove(best_group)
                best_score = best_group_score
                logger.info(f"Добавлена группа '{best_group}': F1 = {best_score:.4f}, "
                            f"Всего признаков = {len(current_features)}")
            else:
                # Если ни одна группа не улучшает результат, выходим из цикла
                break

        # Если после жадного выбора остались неиспользованные группы, добавляем их в конец
        for group in available_groups:
            selected_groups.append(group)
            logger.warning(f"Группа '{group}' добавлена в конец списка без улучшения F1")

        logger.info(f"Оптимизированная последовательность групп: {selected_groups}")
        return selected_groups

    def optimize_feature_groups_order(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            kfold) -> List[str]:
        """
        Оптимизация порядка добавления групп признаков с использованием жадного алгоритма
        """
        if not self.feature_groups:
            self.feature_groups = self.group_features(X_train_val)

        # Создаем отфильтрованную версию feature_groups только с теми признаками, которые действительно присутствуют в X_train_val
        filtered_feature_groups = {}
        for group, features in self.feature_groups.items():
            filtered_features = [f for f in features if f in X_train_val.columns]
            if filtered_features:  # Включаем группу только если в ней есть хотя бы один признак
                filtered_feature_groups[group] = filtered_features

        return self.optimize_feature_groups_order_1(X_train_val, y_train_val, filtered_feature_groups, kfold)

    def select_optimal_imbalance_method(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            kfold) -> Tuple[str, Dict]:
        """
        Выбор оптимального метода обработки дисбаланса с использованием кросс-валидации

        Args:
            X_train_val: DataFrame с признаками
            y_train_val: Серия целевых значений
            kfold: Объект кросс-валидации

        Returns:
            Tuple[str, Dict]: Оптимальный метод и параметры для него
        """
        logger.info("Выбор оптимального метода обработки дисбаланса с кросс-валидацией")

        # Методы для тестирования
        methods = ['none', 'class_weight', 'oversample', 'undersample', 'smote']

        # Базовые параметры LightGBM
        base_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'random_state': self.random_state,
            'verbose': -1
        }

        # Результаты по методам
        method_results = {}

        for method in methods:
            logger.info(f"Оценка метода '{method}'")

            # Инициализация массивов для метрик
            fold_f1_scores = []

            # Перебор фолдов
            for train_idx, val_idx in kfold.split(X_train_val, y_train_val):
                # Разделение данных для текущего фолда
                X_train_fold = X_train_val.iloc[train_idx]
                y_train_fold = y_train_val.iloc[train_idx]
                X_val_fold = X_train_val.iloc[val_idx]
                y_val_fold = y_train_val.iloc[val_idx]

                # Обработка дисбаланса для текущего фолда
                X_train_processed, y_train_processed, imbalance_params = self.handle_class_imbalance(
                    X_train_fold, y_train_fold, method=method)

                # Обновление параметров модели, если необходимо
                current_params = base_params.copy()
                if 'class_weight' in imbalance_params:
                    # Для бинарной классификации используем scale_pos_weight
                    weight_pos = imbalance_params['class_weight'].get(1, 1.0)
                    weight_neg = imbalance_params['class_weight'].get(0, 1.0)
                    current_params['scale_pos_weight'] = weight_neg / weight_pos

                # Обучение модели
                lgb_train = lgb.Dataset(X_train_processed, y_train_processed)
                model = lgb.train(current_params, lgb_train, num_boost_round=100)

                # Оценка на валидационном наборе
                y_pred = model.predict(X_val_fold)
                y_pred_binary = (y_pred > 0.5).astype(int)
                f1 = f1_score(y_val_fold, y_pred_binary)
                fold_f1_scores.append(f1)

            # Средний F1-score для данного метода
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)

            method_results[method] = {
                'mean_f1': mean_f1,
                'std_f1': std_f1
            }

            logger.info(f"Метод '{method}': F1 = {mean_f1:.4f} ± {std_f1:.4f}")

        # Выбор лучшего метода
        best_method = max(method_results.items(), key=lambda x: x[1]['mean_f1'])[0]

        logger.info(
            f"Лучший метод обработки дисбаланса: '{best_method}' с F1 = {method_results[best_method]['mean_f1']:.4f}")

        # Параметры для лучшего метода
        best_params = {}
        if best_method == 'class_weight':
            _, _, best_params = self.handle_class_imbalance(X_train_val, y_train_val, method=best_method)

        return best_method, best_params

    def handle_class_imbalance(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            method: str = 'class_weight',
            random_state: Optional[int] = None,
            sampling_strategy: Union[str, float, Dict] = 'auto',
            k_neighbors: int = 5) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Обработка дисбаланса классов в данных

        Args:
            X_train_val: DataFrame с признаками
            y_train_val: Серия целевых значений
            method: Метод обработки дисбаланса ('class_weight', 'oversample', 'undersample', 'smote', 'adasyn', 'none')
            random_state: Seed для генератора случайных чисел
            sampling_strategy: Стратегия сэмплирования (соотношение классов, словарь, 'auto', 'majority', 'minority', 'not minority')
            k_neighbors: Количество соседей для методов SMOTE и ADASYN

        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict]:
                - Переработанные признаки
                - Переработанные целевые значения
                - Словарь с параметрами для модели
        """
        if random_state is None:
            random_state = self.random_state

        logger.info(f"Обработка дисбаланса классов методом '{method}'")

        # Анализ распределения классов
        class_distribution = y_train_val.value_counts().sort_index()

        if len(class_distribution) <= 1:
            logger.warning("Данные содержат только один класс! Обработка дисбаланса не требуется.")
            return X_train_val, y_train_val, {}

        class_ratio = class_distribution.iloc[0] / class_distribution.iloc[1] if len(class_distribution) > 1 else float(
            'inf')

        logger.info(f"Исходное распределение классов: {class_distribution.to_dict()}")
        logger.info(f"Соотношение классов (отриц./полож.): {class_ratio:.2f}")

        # Параметры для возврата
        imbalance_params = {}

        # Сохранение индексов для возможного восстановления
        original_indices = X_train_val.index

        # Если дисбаланс незначительный или не нужна обработка
        if method == 'none' or (0.5 <= class_ratio <= 2.0):
            logger.info("Дисбаланс классов незначительный или обработка отключена")
            return X_train_val, y_train_val, imbalance_params

        if method == 'class_weight':
            # Расчет весов классов для использования при обучении
            n_samples = len(y_train_val)
            n_classes = len(class_distribution)

            class_weights = {}
            for c in class_distribution.index:
                class_weights[int(c)] = n_samples / (n_classes * class_distribution[c])

            logger.info(f"Рассчитаны веса классов: {class_weights}")
            imbalance_params['class_weight'] = class_weights

            # Возвращаем исходные данные и параметры
            return X_train_val, y_train_val, imbalance_params

        elif method == 'oversample':
            try:
                # Применение RandomOverSampler
                oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
                X_resampled_np, y_resampled_np = oversampler.fit_resample(X_train_val.values, y_train_val.values)

                # Преобразование обратно в DataFrame и Series
                X_resampled = pd.DataFrame(X_resampled_np, columns=X_train_val.columns)
                y_resampled = pd.Series(y_resampled_np, name=y_train_val.name)

                new_class_dist = pd.Series(y_resampled).value_counts().sort_index()
                logger.info(f"Выполнен оверсэмплинг. Новое распределение: {new_class_dist.to_dict()}")
                return X_resampled, y_resampled, imbalance_params

            except ImportError:
                logger.error(
                    "Для метода 'oversample' требуется библиотека imbalanced-learn. Используйте 'pip install imbalanced-learn'")
                return X_train_val, y_train_val, imbalance_params

        elif method == 'undersample':
            try:
                # Применение RandomUnderSampler
                undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
                X_resampled_np, y_resampled_np = undersampler.fit_resample(X_train_val.values, y_train_val.values)

                # Преобразование обратно в DataFrame и Series
                X_resampled = pd.DataFrame(X_resampled_np, columns=X_train_val.columns)
                y_resampled = pd.Series(y_resampled_np, name=y_train_val.name)

                new_class_dist = pd.Series(y_resampled).value_counts().sort_index()
                logger.info(f"Выполнен андерсэмплинг. Новое распределение: {new_class_dist.to_dict()}")
                return X_resampled, y_resampled, imbalance_params

            except ImportError:
                logger.error(
                    "Для метода 'undersample' требуется библиотека imbalanced-learn. Используйте 'pip install imbalanced-learn'")
                return X_train_val, y_train_val, imbalance_params

        elif method == 'smote':
            try:
                # Применение SMOTE
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)
                X_resampled_np, y_resampled_np = smote.fit_resample(X_train_val.values, y_train_val.values)

                # Преобразование обратно в DataFrame и Series
                X_resampled = pd.DataFrame(X_resampled_np, columns=X_train_val.columns)
                y_resampled = pd.Series(y_resampled_np, name=y_train_val.name)

                new_class_dist = pd.Series(y_resampled).value_counts().sort_index()
                logger.info(f"Выполнен SMOTE. Новое распределение: {new_class_dist.to_dict()}")
                return X_resampled, y_resampled, imbalance_params

            except ImportError:
                logger.error(
                    "Для метода 'smote' требуется библиотека imbalanced-learn. Используйте 'pip install imbalanced-learn'")
                return X_train_val, y_train_val, imbalance_params

        elif method == 'adasyn':
            try:
                # Применение ADASYN
                adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state, n_neighbors=k_neighbors)
                X_resampled_np, y_resampled_np = adasyn.fit_resample(X_train_val.values, y_train_val.values)

                # Преобразование обратно в DataFrame и Series
                X_resampled = pd.DataFrame(X_resampled_np, columns=X_train_val.columns)
                y_resampled = pd.Series(y_resampled_np, name=y_train_val.name)

                new_class_dist = pd.Series(y_resampled).value_counts().sort_index()
                logger.info(f"Выполнен ADASYN. Новое распределение: {new_class_dist.to_dict()}")
                return X_resampled, y_resampled, imbalance_params

            except ImportError:
                logger.error(
                    "Для метода 'adasyn' требуется библиотека imbalanced-learn. Используйте 'pip install imbalanced-learn'")
                return X_train_val, y_train_val, imbalance_params

        else:
            logger.warning(f"Неизвестный метод обработки дисбаланса: {method}. Используем исходные данные.")
            return X_train_val, y_train_val, imbalance_params

    def analyze_class_distribution(self, y: pd.Series) -> Dict:
        """
        Анализ распределения классов в данных

        Args:
            y: Серия целевых значений

        Returns:
            Dict: Словарь со статистикой распределения классов
        """
        logger.info("Анализ распределения классов")

        # Подсчет количества каждого класса
        class_counts = y.value_counts().sort_index()

        # Доли классов
        class_proportions = class_counts / len(y)

        # Вычисление метрик сбалансированности
        n_classes = len(class_counts)

        if n_classes <= 1:
            logger.warning("Обнаружен только один класс в данных!")
            imbalance_ratio = float('inf')
            entropy = 0
        else:
            # Соотношение дисбаланса (соотношение между наибольшим и наименьшим классами)
            imbalance_ratio = class_counts.max() / class_counts.min()

            # Энтропия распределения классов (мера равномерности)
            entropy = -sum(p * np.log2(p) for p in class_proportions if p > 0)

        # Расчет теоретически идеальной энтропии
        ideal_entropy = np.log2(n_classes) if n_classes > 0 else 0

        # Индекс равномерности (1 = идеально равномерное, 0 = крайне несбалансированное)
        uniformity_index = entropy / ideal_entropy if ideal_entropy > 0 else 0

        stats = {
            'n_classes': n_classes,
            'class_counts': class_counts.to_dict(),
            'class_proportions': class_proportions.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'entropy': entropy,
            'ideal_entropy': ideal_entropy,
            'uniformity_index': uniformity_index
        }

        logger.info(f"Количество классов: {n_classes}")
        logger.info(f"Соотношение дисбаланса: {imbalance_ratio:.2f}")
        logger.info(f"Индекс равномерности: {uniformity_index:.2f} (1 = идеально равномерно)")

        return stats

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
