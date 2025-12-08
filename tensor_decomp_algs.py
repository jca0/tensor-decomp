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

def tucker_als(T, k, max_iter=50, tol=1e-4):
    """
    Tucker ALS decomposition for 3D tensors.
    Decomposes T ≈ S ×₁ U ×₂ V ×₃ W where S is (k,k,k) core tensor.

    Args:
        T: d x d x d tensor
        k: rank for each mode (core will be k x k x k)
        max_iter: number of ALS iterations
        tol: stopping tolerance (optional)

    Returns:
        U, V, W: factor matrices (d x k each)
        S: core tensor (k x k x k)
    """
    d = T.shape[0]

    # Initialize factor matrices
    U = np.random.randn(d, k)
    V = np.random.randn(d, k)
    W = np.random.randn(d, k)

    # Normalize columns
    def normalize(X):
        return X / np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-10)

    U = normalize(U)
    V = normalize(V)
    W = normalize(W)

    # Initialize core tensor
    S = np.random.randn(k, k, k)

    for it in range(max_iter):
        # Update core tensor S first
        T1 = T.reshape(d, d*d)
        kr_vw = khatri_rao(V, W)  # (d*d) x k
        S_mode1_new = U.T @ T1 @ kr_vw
        S = S_mode1_new.reshape(k, k, k)

        # Update U (mode 1)
        S_mode1 = S.reshape(k, k*k)  # unfold S along mode 1
        Y = T1 @ kr_vw @ S_mode1.T
        U = Y @ np.linalg.pinv(S_mode1 @ S_mode1.T)
        U = normalize(U)

        # Update V (mode 2)
        T2 = np.transpose(T, (1, 0, 2)).reshape(d, d*d)
        kr_uw = khatri_rao(U, W)
        S_mode2 = np.transpose(S, (1, 0, 2)).reshape(k, k*k)
        Y = T2 @ kr_uw @ S_mode2.T
        V = Y @ np.linalg.pinv(S_mode2 @ S_mode2.T)
        V = normalize(V)

        # Update W (mode 3)
        T3 = np.transpose(T, (2, 0, 1)).reshape(d, d*d)
        kr_vu = khatri_rao(U, V)
        S_mode3 = np.transpose(S, (2, 0, 1)).reshape(k, k*k)
        Y = T3 @ kr_vu @ S_mode3.T
        W = Y @ np.linalg.pinv(S_mode3 @ S_mode3.T)
        W = normalize(W)

    return U, V, W, S