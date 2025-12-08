from utils import khatri_rao
import numpy as np
from scipy.linalg import eig, pinv

def jennrich(T_noisy, k):
    d = T_noisy.shape[0]
    u = np.random.randn(d)
    v = np.random.randn(d)
    M1 = np.sum(u[:, np.newaxis, np.newaxis] * T_noisy, axis=2)
    M2 = np.sum(v[:, np.newaxis, np.newaxis] * T_noisy, axis=2)
    try:
        vals, vl, vr = eig(M1, M2, left=True, right=True)
        if np.any(np.imag(vals) != 0):
            return None, None, None
        vl = np.real(vl)
        vr = np.real(vr)
        A_hat = np.linalg.inv(vl).T
        B_hat = np.linalg.inv(vr).T
        T3 = np.transpose(T_noisy, (2, 0, 1)).reshape(d, d * d)
        kr = khatri_rao(A_hat, B_hat)
        C_hat = T3 @ pinv(kr.T)
        return A_hat, B_hat, C_hat
    except:
        return None, None, None

def cp_als(T, k, max_iter=50, tol=1e-4):
    d = T.shape[0]
    A = np.random.randn(d, k)
    B = np.random.randn(d, k)
    C = np.random.randn(d, k)
    norm = lambda X: np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-10)
    A = A / norm(A)
    B = B / norm(B)
    C = C / norm(C)
    for _ in range(max_iter):
        # Update A
        kr = khatri_rao(B, C)
        T1 = T.reshape(d, d * d)
        A = T1 @ kr @ pinv(kr.T @ kr)
        A = A / norm(A)
        # Update B
        kr = khatri_rao(A, C)
        T2 = np.transpose(T, (1, 0, 2)).reshape(d, d * d)
        B = T2 @ kr @ pinv(kr.T @ kr)
        B = B / norm(B)
        # Update C
        kr = khatri_rao(A, B)
        T3 = np.transpose(T, (2, 0, 1)).reshape(d, d * d)
        C = T3 @ kr @ pinv(kr.T @ kr)
        C = C / norm(C)
    return A, B, C

def ortho_als(T, k, max_iter=50, tol=1e-4):
    """
    Ortho-ALS for CP decomposition.
    Adds an orthogonalization (QR) step after each factor update.

    Args:
        T: d x d x d tensor
        k: rank
        max_iter: number of ALS iterations
        tol: stopping tolerance (optional, usually not needed)

    Returns:
        A, B, C: factor matrices (d x k)
    """
    d = T.shape[0]

    # Random initialization
    A = np.random.randn(d, k)
    B = np.random.randn(d, k)
    C = np.random.randn(d, k)

    # Normalize columns
    def normalize(X):
        return X / np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-10)

    A = normalize(A)
    B = normalize(B)
    C = normalize(C)

    # Unfoldings
    T1 = T.reshape(d, d * d)
    T2 = np.transpose(T, (1, 0, 2)).reshape(d, d * d)
    T3 = np.transpose(T, (2, 0, 1)).reshape(d, d * d)

    for it in range(max_iter):

        # --- Update A ---
        kr = khatri_rao(B, C)        # (d*d) x k
        A = T1 @ kr @ pinv(kr.T @ kr)
        # Ortho step: QR with economic mode
        A, _ = np.linalg.qr(A)
        A = A[:, :k]                 # take first k orthonormal cols
        A = normalize(A)

        # --- Update B ---
        kr = khatri_rao(A, C)
        B = T2 @ kr @ pinv(kr.T @ kr)
        B, _ = np.linalg.qr(B)
        B = B[:, :k]
        B = normalize(B)

        # --- Update C ---
        kr = khatri_rao(A, B)
        C = T3 @ kr @ pinv(kr.T @ kr)
        C, _ = np.linalg.qr(C)
        C = C[:, :k]
        C = normalize(C)

    return A, B, C