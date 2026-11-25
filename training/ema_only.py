"""仅使用历史产量的指数移动平均。默认系数为 0.5，无需拟合网络。"""


MODEL_ID = 'ema_only'


def predict_history(panel, test_indices):
    """Return the precomputed, strictly historical EMA and aligned test labels."""
    assert (panel['s'][test_indices, -1] == 0).all(), 'EMA-only missing historical values'
    return panel['emas'][test_indices, 1], panel['y'][test_indices], panel['counties'][test_indices]



if __name__ == '__main__':
    from _shared.train import main

    main(MODEL_ID)
