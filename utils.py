import numpy as np
from scipy.optimize import linear_sum_assignment

def khatri_rao(P, Q):
    d1, k = P.shape
    d2, _ = Q.shape
    return np.hstack([np.kron(P[:, i], Q[:, i])[:, np.newaxis] for i in range(k)])

def reconstruct_tensor(A, B, C):
    d, k = A.shape
    T = np.zeros((d, d, d))
    for i in range(k):
        T += np.outer(A[:, i], B[:, i])[:, :, np.newaxis] * C[:, i]
    return T

def fro_norm(X):
    return np.sqrt(np.sum(X**2))

def reconstruction_error(true_T, hat_T):
    return fro_norm(true_T - hat_T) / fro_norm(true_T)

def factor_recovery_error(true_F, hat_F):
    d, k = true_F.shape
    true_norm = true_F / np.linalg.norm(true_F, axis=0)
    hat_norm = hat_F / np.linalg.norm(hat_F, axis=0)
    cos_matrix = np.abs(true_norm.T @ hat_norm)
    cost = -cos_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    mean_cos = np.mean(cos_matrix[row_ind, col_ind])
    return mean_cos  # Higher is better; for error, could use 1 - mean_cos if desired