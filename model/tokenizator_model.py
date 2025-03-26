import os
import numpy as np
import optuna
import pandas as pd
import logging
import lightgbm as lgb
from typing import Tuple, List, Dict, Optional, Union

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score
from .preprocessing import preprocess_data
from .feature_selection import analyze_correlations, group_features
from .model_training import train_model, predict, split_data
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

        X_reduced, to_drop = analyze_correlations(X, threshold)
        logger.info(f"Осталось признаков после удаления коррелирующих: {X_reduced.shape[1]} (было {X.shape[1]})")

        return X_reduced, to_drop


    def select_features_with_optimal_parameters(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            params: Dict = None,
            group_weights: Dict[str, float] = None,
            trial_budget: int = 100
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Выбор признаков с учетом оптимальных параметров и группового подхода.

        Args:
            X_train_val: DataFrame с признаками
            y_train_val: Серия целевых значений
            params: Гиперпараметры LightGBM (используются при оценке признаков)
            group_weights: Словарь весов для групп признаков (влияет на вероятность выбора)
            trial_budget: Количество итераций поиска оптимальной комбинации

        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame с выбранными признаками и их список
        """

        logger.info(f"Начало отбора признаков с учетом оптимальных параметров (бюджет: {trial_budget} итераций)")

        # Если группы признаков не определены, группируем их
        if not self.feature_groups:
            self.feature_groups = self.group_features(X_train_val)

        # Если не переданы веса групп, используем равномерное распределение
        if group_weights is None:
            group_weights = {group: 1.0 for group in self.feature_groups.keys()}

        # Если не переданы параметры, используем дефолтные
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
                    'random_state': self.random_state
                }

        # Создаем кросс-валидацию
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Объект для оптимизации
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        # Определение объективной функции для оптимизации
        def objective(trial):
            # 1. Для каждой группы признаков определяем, будет ли она использоваться
            # С учетом весов групп (вероятность использования пропорциональна весу)
            selected_groups = []
            for group_name, weight in group_weights.items():
                # Нормализуем вес к вероятности (от 0 до 1)
                prob = min(1.0, max(0.1, weight / max(group_weights.values())))

                random_value = trial.suggest_float(f"prob_{group_name}", 0.0, 1.0)
                use_group = random_value < prob

                if use_group:
                    selected_groups.append(group_name)

            # Если не выбрана ни одна группа, используем случайную с учетом весов
            if not selected_groups and self.feature_groups:
                group_probs = np.array(list(group_weights.values()))
                group_probs = group_probs / group_probs.sum()
                selected_groups = [np.random.choice(list(group_weights.keys()), p=group_probs)]

            # 2. Сбор всех признаков из выбранных групп
            candidate_features = []
            for group in selected_groups:
                candidate_features.extend(self.feature_groups[group])

            # 3. Оптимизация подмножества признаков внутри каждой группы
            selected_features = []
            for group in selected_groups:
                group_features = self.feature_groups[group]
                # Определяем сколько признаков из группы использовать
                feature_ratio = trial.suggest_float(f"feature_ratio_{group}", 0.3, 1.0)
                n_features = max(1, int(len(group_features) * feature_ratio))

                # Если признаков мало, используем все
                if len(group_features) <= 5:
                    selected_features.extend(group_features)
                else:
                    # Иначе выбираем подмножество признаков
                    # Здесь мы используем случайное подмножество для скорости
                    # В реальной системе можно использовать более сложные методы отбора
                    selected_indices = np.random.choice(
                        len(group_features),
                        size=n_features,
                        replace=False
                    )
                    selected_features.extend([group_features[i] for i in selected_indices])

            # Если отобрано слишком много признаков, ограничиваем их число
            max_features = min(200, len(selected_features))
            if len(selected_features) > max_features:
                selected_indices = np.random.choice(
                    len(selected_features),
                    size=max_features,
                    replace=False
                )
                selected_features = [selected_features[i] for i in selected_indices]

            # 4. Оценка полученного набора признаков с помощью кросс-валидации
            try:
                # Обучаем модель с выбранными признаками и оптимальными параметрами
                model = lgb.LGBMClassifier(**params)
                cv_scores = cross_val_score(
                    model,
                    X_train_val[selected_features],
                    y_train_val,
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1
                )

                mean_score = np.mean(cv_scores)

                # Штраф за большое количество признаков для баланса между качеством и простотой
                n_features_penalty = 1.0 - (0.0005 * len(selected_features))
                score = mean_score * n_features_penalty

                return score
            except Exception as e:
                logger.error(f"Ошибка при оценке признаков: {e}")
                return 0.0

        # Запуск оптимизации
        study.optimize(objective, n_trials=trial_budget, n_jobs=-1)

        # Получение лучших параметров
        best_trial = study.best_trial

        # Восстановление выбранных групп признаков из лучшего решения
        selected_groups = []
        for group_name, weight in group_weights.items():
            prob = min(1.0, max(0.1, weight / max(group_weights.values())))
            random_value = best_trial.params.get(f"prob_{group_name}", 0.5)
            if random_value < prob:
                selected_groups.append(group_name)

        # Если не выбрана ни одна группа, используем все группы
        if not selected_groups and self.feature_groups:
            selected_groups = list(self.feature_groups.keys())

        # Составление итогового списка признаков
        final_features = []
        for group in selected_groups:
            group_features = self.feature_groups[group]
            # Определяем сколько признаков из группы использовать
            feature_ratio = best_trial.params.get(f"feature_ratio_{group}", 1.0)
            n_features = max(1, int(len(group_features) * feature_ratio))

            if len(group_features) <= 5 or feature_ratio >= 0.95:
                # Если признаков мало или выбрана почти вся группа, используем все признаки
                final_features.extend(group_features)
            else:
                # Иначе выбираем топ-N признаков по важности
                # Обучаем модель на всех признаках группы
                temp_model = lgb.LGBMClassifier(**params)
                temp_model.fit(X_train_val[group_features], y_train_val)

                # Получаем важность признаков
                importance = pd.DataFrame({
                    'feature': group_features,
                    'importance': temp_model.feature_importances_
                }).sort_values('importance', ascending=False)

                # Выбираем топ-N признаков
                top_features = importance.head(n_features)['feature'].tolist()
                final_features.extend(top_features)

        logger.info(f"Отбор признаков завершен. Выбрано {len(final_features)} признаков "
                    f"из {len(selected_groups)} групп.")

        # Сохраняем выбранные признаки
        self.selected_features = final_features

        return X_train_val[final_features], final_features


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

    def train_final_model(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            params: Optional[Dict] = None) -> None:
        """
        Переобучение финальной модели на тренировочных и валидационных данных.

        Args:
            X_train_val: DataFrame с признаками обучающей и валидационной выборки
            y_train_val: Серия целевых значений обучающей и валидационной выборки
            params: Параметры модели LightGBM
        """
        logger.info("Переобучение финальной модели на тренировочных и валидационных данных")

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
        self.feature_names = X_train_val.columns.tolist()

        # Создание датасета для обучения
        train_data = lgb.Dataset(X_train_val, label=y_train_val, feature_name=self.feature_names)

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

        logger.info(f"Финальная модель обучена на {len(X_train_val)} примерах с {len(self.feature_names)} признаками")

    def optimize_jointly(
            self,
            X_train_val: pd.DataFrame,
            y_train_val: pd.Series,
            cv_outer: int = 3,
            cv_inner: int = 5,
            n_trials: int = 30
    ) -> Tuple[Dict, List[str]]:
        """
        Совместная оптимизация гиперпараметров и отбора признаков с использованием
        вложенной кросс-валидации.

        Args:
            X_train_val: DataFrame с признаками обучающей и валидационной выборки
            y_train_val: Серия целевых значений обучающей и валидационной выборки
            cv_outer: Количество фолдов для внешней кросс-валидации
            cv_inner: Количество фолдов для внутренней кросс-валидации
            n_trials: Количество испытаний Optuna для каждого фолда внешней кросс-валидации

        Returns:
            Tuple[Dict, List[str]]: Оптимальные гиперпараметры и список выбранных признаков
        """

        logger.info(f"Запуск совместной оптимизации с вложенной кросс-валидацией "
                    f"({cv_outer} внешних фолдов, {cv_inner} внутренних фолдов, {n_trials} испытаний)")

        # Если у нас нет сгруппированных признаков, сначала группируем их
        if not self.feature_groups:
            self.feature_groups = self.group_features(X_train_val)

        # Создаем внешнюю кросс-валидацию для оценки стабильности результатов
        outer_cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=self.random_state)

        # Словари для хранения результатов по каждому фолду
        fold_best_params = {}
        fold_best_features = {}
        fold_best_scores = {}

        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_val, y_train_val)):
            logger.info(f"Внешний фолд {fold_idx + 1}/{cv_outer}")

            # Разделение данных для текущего фолда
            X_train_fold = X_train_val.iloc[train_idx]
            y_train_fold = y_train_val.iloc[train_idx]
            X_val_fold = X_train_val.iloc[val_idx]
            y_val_fold = y_train_val.iloc[val_idx]

            # Создаем внутреннюю кросс-валидацию для оптимизации
            inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=self.random_state)

            # Функция для создания объекта исследования Optuna
            def create_study():
                study_optuna = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=self.random_state)
                )
                return study_optuna

            # Объект для оптимизации
            study = create_study()

            # Определение объективной функции для оптимизации
            def objective(trial):
                # 1. Выбор используемых групп признаков
                selected_groups = []
                for group_name in self.feature_groups.keys():
                    use_group = trial.suggest_categorical(f"use_group_{group_name}", [True, False])
                    if use_group:
                        selected_groups.append(group_name)

                # Если не выбрана ни одна группа, используем хотя бы одну случайную
                if not selected_groups and self.feature_groups:
                    selected_groups = [next(iter(self.feature_groups.keys()))]

                # 2. Сбор всех признаков из выбранных групп
                selected_features = []
                for group in selected_groups:
                    selected_features.extend(self.feature_groups[group])

                # Если количество признаков слишком велико, применим дополнительный отбор
                max_features = trial.suggest_int("max_features", min(50, len(selected_features)),
                                                 min(200, len(selected_features)))

                # Если выбрано слишком много признаков, отберем самые важные
                if len(selected_features) > max_features:
                    # Для отбора используем простую модель с дефолтными параметрами и feature importance
                    temp_model = lgb.LGBMClassifier(random_state=self.random_state)
                    temp_model.fit(X_train_fold[selected_features], y_train_fold)

                    # Получение важности признаков
                    feature_importance = pd.DataFrame({
                        'feature': selected_features,
                        'importance': temp_model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    # Отбор топ-N признаков
                    selected_features = feature_importance.head(max_features)['feature'].tolist()

                # 3. Оптимизация гиперпараметров LightGBM
                lgb_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 15, 41),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 10.0, log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                    'random_state': self.random_state,
                    'verbose': -1
                }

                # 4. Оптимальный порог классификации
                threshold = trial.suggest_float('threshold', 0.1, 0.9)

                # 5. Кросс-валидация на внутренних фолдах
                cv_scores = []
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_fold, y_train_fold):
                    # Разделение данных для внутреннего фолда
                    X_inner_train = X_train_fold.iloc[inner_train_idx][selected_features]
                    y_inner_train = y_train_fold.iloc[inner_train_idx]
                    X_inner_val = X_train_fold.iloc[inner_val_idx][selected_features]
                    y_inner_val = y_train_fold.iloc[inner_val_idx]

                    # Обучение модели на внутреннем тренировочном наборе
                    train_data = lgb.Dataset(X_inner_train, label=y_inner_train)
                    val_data = lgb.Dataset(X_inner_val, label=y_inner_val, reference=train_data)

                    callbacks = [lgb.early_stopping(20, verbose=False)]

                    model = lgb.train(
                        lgb_params,
                        train_data,
                        valid_sets=[val_data],
                        callbacks=callbacks,
                        num_boost_round=500
                    )

                    # Оценка на внутреннем валидационном наборе
                    y_pred_proba = model.predict(X_inner_val)
                    y_pred = (y_pred_proba > threshold).astype(int)
                    score = f1_score(y_inner_val, y_pred)
                    cv_scores.append(score)

                # Среднее значение F1 по всем внутренним фолдам
                mean_score = np.mean(cv_scores)

                # Добавление штрафа за большое количество признаков для баланса между качеством и простотой
                feature_penalty = 1.0 - (0.0001 * len(selected_features))
                adjusted_score = mean_score * feature_penalty

                return adjusted_score

            # Запуск оптимизации для текущего внешнего фолда
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)

            # Получение лучших параметров для текущего фолда
            best_trial = study.best_trial
            best_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'random_state': self.random_state,
                'verbose': -1
            }

            # Извлечение оптимальных гиперпараметров LightGBM
            for param_name in ['num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction',
                               'bagging_freq', 'lambda_l1', 'lambda_l2', 'min_child_samples', 'max_depth']:
                if param_name in best_trial.params:
                    best_params[param_name] = best_trial.params[param_name]

            # Извлечение оптимального порога
            best_threshold = best_trial.params.get('threshold', 0.5)

            # Извлечение оптимальных групп признаков
            selected_groups = []
            for group_name in self.feature_groups.keys():
                if best_trial.params.get(f"use_group_{group_name}", False):
                    selected_groups.append(group_name)

            # Если не выбрана ни одна группа, используем все группы
            if not selected_groups and self.feature_groups:
                selected_groups = list(self.feature_groups.keys())

            # Сбор всех признаков из выбранных групп
            selected_features = []
            for group in selected_groups:
                selected_features.extend(self.feature_groups[group])

            # Если существует ограничение на количество признаков, применяем его
            max_features = best_trial.params.get("max_features", len(selected_features))
            if len(selected_features) > max_features:
                # Обучаем модель для определения важности признаков
                temp_model = lgb.LGBMClassifier(random_state=self.random_state)
                temp_model.fit(X_train_fold[selected_features], y_train_fold)

                # Получение важности признаков
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': temp_model.feature_importances_
                }).sort_values('importance', ascending=False)

                # Отбор топ-N признаков
                selected_features = feature_importance.head(max_features)['feature'].tolist()

            # Оценка на внешнем валидационном наборе с оптимальными параметрами
            # Создаем копию лучших параметров без random_state, который будет добавлен в конструкторе
            params_without_random_state = {k: v for k, v in best_params.items() if k != 'random_state'}
            model = lgb.LGBMClassifier(random_state=self.random_state, **params_without_random_state)
            model.fit(X_train_fold[selected_features], y_train_fold)

            y_pred_proba = model.predict_proba(X_val_fold[selected_features])[:, 1]
            y_pred = (y_pred_proba > best_threshold).astype(int)
            val_score = f1_score(y_val_fold, y_pred)

            # Сохранение результатов для текущего фолда
            fold_best_params[fold_idx] = best_params
            fold_best_params[fold_idx]['threshold'] = best_threshold
            fold_best_features[fold_idx] = selected_features
            fold_best_scores[fold_idx] = val_score

            logger.info(f"Внешний фолд {fold_idx + 1} - Лучший F1: {val_score:.4f}, "
                        f"Количество признаков: {len(selected_features)}, "
                        f"Выбранные группы: {selected_groups}")

        # Выбор лучшего фолда на основе валидационного F1-score
        best_fold = max(fold_best_scores.items(), key=lambda x: x[1])[0]

        logger.info(f"Лучший результат в фолде {best_fold + 1} с F1 = {fold_best_scores[best_fold]:.4f}")

        # Сохранение оптимальных параметров и признаков
        self.best_params = fold_best_params[best_fold]
        self.best_threshold = self.best_params.pop('threshold', 0.5)
        self.selected_features = fold_best_features[best_fold]

        # Обучение итоговой модели на всех тренировочных данных
        # с оптимальными параметрами и признаками
        logger.info(f"Обучение промежуточной модели с оптимальными параметрами "
                    f"на {len(self.selected_features)} признаках")

        # Создаем копию лучших параметров без random_state, который будет добавлен в конструкторе
        params_without_random_state = {k: v for k, v in self.best_params.items() if k != 'random_state'}
        self.model = lgb.LGBMClassifier(random_state=self.random_state, **params_without_random_state)
        self.model.fit(X_train_val[self.selected_features], y_train_val)

        # Расчет важности признаков
        self.feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Резюме для логирования
        logger.info(f"Завершена совместная оптимизация. Выбрано {len(self.selected_features)} признаков.")
        logger.info(f"Оптимальный порог классификации: {self.best_threshold:.4f}")
        logger.info(f"Оптимальные параметры: {self.best_params}")

        return self.best_params, self.selected_features


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
        # methods = ['none', 'class_weight', 'oversample', 'undersample', 'smote']
        methods = ['class_weight']

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
                    current_params['scale_pos_weight'] = weight_pos / weight_neg

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
