import numpy as np
from sklearn.metrics import log_loss


def pri_matrix_loss(y_true:np.ndarray, y_pred:np.ndarray):
    return log_loss(y_true.flatten(), y_pred.flatten())  # * y_true.shape[1]
