"""随机森林。汇总 9 项日变量的生长季均值与标准差，并使用 8 项静态输入。"""


from itertools import product
import pickle
import random
import time

import numpy as np

from _shared.dataset import split_indices
from _shared.protocol import TUNING_SEED, TUNING_YEAR


MODEL_ID = 'rf'


SEARCH_SPACE = {
    "n_estimators": (100, 200, 400),
    "max_depth": (5, 10, 15, None),
    "min_samples_leaf": (1, 2, 5),
    "max_features": ("sqrt", 0.7, 1.0),
}


def candidate_configs(trials, seed):
    grid = list(product(*SEARCH_SPACE.values()))
    if not 1 <= trials <= len(grid):
        raise ValueError(f"Select between 1 and {len(grid)} trial configurations")
    rng = random.Random(seed)
    return [dict(zip(SEARCH_SPACE, values)) for values in rng.sample(grid, trials)]



def _aggregate_rf_features(x: np.ndarray) -> np.ndarray:
    """把时序特征压缩成固定长度，供传统机器学习模型使用。"""
    # 用均值和标准差概括整条时间序列
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return np.concatenate([mean, std], axis=1)


def fit_rf(config, x, y, seed):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(**config, random_state=seed, n_jobs=1)
    started = time.perf_counter()
    model.fit(x, y)
    return model, time.perf_counter() - started


def rf_features(panel, ids):
    return np.concatenate(
        [_aggregate_rf_features(panel['x'][ids, :, :9]), panel['s'][ids]], axis=1,
    )


def tune_rf(panel, metric):
    train, validation, _ = split_indices(panel, TUNING_YEAR)
    x_train = rf_features(panel, train)
    x_validation = rf_features(panel, validation)
    trials = []
    for config in candidate_configs(20, TUNING_SEED):
        model, _ = fit_rf(config, x_train, panel['y'][train], TUNING_SEED)
        rmse = metric(panel['y'][validation], model.predict(x_validation))['rmse']
        trials.append({'config': config, 'validation_rmse': rmse})
    winner = min(trials, key=lambda trial: trial['validation_rmse'])
    return {'selected_config': winner['config'], 'trials': trials,
            'validation_years': sorted(set(panel['years'][validation].tolist()))}


def refit_rf(panel, refit, test, config, seed, checkpoint):
    model, seconds = fit_rf(config, rf_features(panel, refit), panel['y'][refit], seed)
    with checkpoint.open('wb') as stream:
        pickle.dump(model, stream)
    inputs = rf_features(panel, test)
    predictions = model.predict(inputs)
    with checkpoint.open('rb') as stream:
        restored = pickle.load(stream).predict(inputs)
    np.testing.assert_array_equal(predictions, restored)
    return predictions, seconds


if __name__ == '__main__':
    from _shared.train import main

    main(MODEL_ID)
