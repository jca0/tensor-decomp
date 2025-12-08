import numpy as np
from scipy.optimize import linear_sum_assignment

def khatri_rao(P, Q):
    """
    Khatri-Rao product (block Kronecker product)
    """
    d1, k = P.shape
    d2, _ = Q.shape
    return np.hstack([np.kron(P[:, i], Q[:, i])[:, np.newaxis] for i in range(k)])

def reconstruct_tensor(A, B, C):
    """
    Reconstruct tensor from factor matrices via CP decomposition.
    """
    d, k = A.shape
    T = np.zeros((d, d, d))
    for i in range(k):
        T += np.outer(A[:, i], B[:, i])[:, :, np.newaxis] * C[:, i]
    return T

def reconstruct_tensor_tucker(U, V, W, S):
    """
    Reconstruct tensor from Tucker decomposition: T ≈ S ×₁ U ×₂ V ×₃ W
    """
    d, r1 = U.shape
    _, r2 = V.shape
    _, r3 = W.shape
    T = np.zeros((d, d, d))

    # For efficiency, we can use einstein summation
    # T[i,j,k] = sum_{p,q,r} S[p,q,r] * U[i,p] * V[j,q] * W[k,r]
    T = np.einsum('pqr,ip,jq,kr->ijk', S, U, V, W)
    return T

def fro_norm(X):
    """
    Frobenius norm
    """
    return np.sqrt(np.sum(X**2))

def reconstruction_error(true_T, hat_T):
    return fro_norm(true_T - hat_T) / fro_norm(true_T)

def factor_recovery_error(true_F, hat_F):
    """
    Factor recovery error by matching the factor matrices up to a rotation.
    """
    d, k = true_F.shape
    true_norm = true_F / np.linalg.norm(true_F, axis=0)
    hat_norm = hat_F / np.linalg.norm(hat_F, axis=0)
    cos_matrix = np.abs(true_norm.T @ hat_norm)
    cost = -cos_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    mean_cos = np.mean(cos_matrix[row_ind, col_ind])
    return mean_cos  # Higher is better; for error, could use 1 - mean_cos if desired