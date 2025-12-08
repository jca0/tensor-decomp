import numpy as np
from data_gen import *
from utils import reconstruct_tensor, fro_norm, factor_recovery_error, reconstruction_error
from tensor_decomp_algs import jennrich, cp_als, ortho_als
from tqdm import tqdm
import matplotlib.pyplot as plt

# Experiment 1: Bad Conditioning Test
def experiment1(d=20, k=3, sigma=0.1, num_reps=50):
    rhos = np.linspace(0, 0.99, 10)
    jenn_rec = []
    als_rec = []
    ortho_rec = []
    jenn_fac = []
    als_fac = []
    ortho_fac = []
    for rho in tqdm(rhos):
        rec_j, fac_j, rec_a, fac_a, rec_o, fac_o = [], [], [], [], [], []
        for _ in tqdm(range(num_reps)):
            T, T_noisy, A_true, B_true, C_true = generate_data_gmm(d, k, sigma, rho)
            # Jennrich
            A_hat, B_hat, C_hat = jennrich(T_noisy, k)
            if A_hat is not None:
                hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
                rec_j.append(reconstruction_error(T, hat_T))
                fac_j.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
            # ALS
            A_hat, B_hat, C_hat = cp_als(T_noisy, k)
            hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
            rec_a.append(reconstruction_error(T, hat_T))
            fac_a.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
            # Ortho-ALS
            A_hat, B_hat, C_hat = ortho_als(T_noisy, k)
            hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
            rec_o.append(reconstruction_error(T, hat_T))
            fac_o.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
        jenn_rec.append(np.mean(rec_j) if rec_j else np.nan)
        als_rec.append(np.mean(rec_a))
        ortho_rec.append(np.mean(rec_o))
        jenn_fac.append(np.mean(fac_j) if fac_j else np.nan)
        als_fac.append(np.mean(fac_a))
        ortho_fac.append(np.mean(fac_o))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rhos, jenn_rec, label='Jennrich Rec Err')
    ax1.plot(rhos, als_rec, label='ALS Rec Err')
    ax1.plot(rhos, ortho_rec, label='Ortho-ALS Rec Err')
    ax1.legend()
    ax1.set_xlabel('rho')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('Reconstruction Error vs rho')

    ax2.plot(rhos, jenn_fac, label='Jennrich Fac Sim')
    ax2.plot(rhos, als_fac, label='ALS Fac Sim')
    ax2.plot(rhos, ortho_fac, label='Ortho-ALS Fac Sim')
    ax2.legend()
    ax2.set_xlabel('rho')
    ax2.set_ylabel('Avg Cosine Similarity')
    ax2.set_title('Factor Similarity vs rho')

    plt.tight_layout()
    plt.show()

# Experiment 2: Noise Floor
def experiment2(d=20, k=3, rho=0, num_reps=50):
    sigmas = np.logspace(-3, 0, 10)
    jenn_rec = []
    als_rec = []
    ortho_rec = []
    jenn_fac = []
    als_fac = []
    ortho_fac = []
    for sigma in sigmas:
        rec_j, fac_j, rec_a, fac_a = [], [], [], [] 
        rec_o, fac_o = [], []
        for _ in range(num_reps):
            T, T_noisy, A_true, B_true, C_true = generate_data_gmm(d, k, sigma, rho)
            A_hat, B_hat, C_hat = jennrich(T_noisy, k)
            if A_hat is not None:
                hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
                rec_j.append(reconstruction_error(T, hat_T))
                fac_j.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
            A_hat, B_hat, C_hat = cp_als(T_noisy, k)
            hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
            rec_a.append(reconstruction_error(T, hat_T))
            fac_a.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
            A_hat, B_hat, C_hat = ortho_als(T_noisy, k)
            hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
            rec_o.append(reconstruction_error(T, hat_T))
            fac_o.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
        jenn_rec.append(np.mean(rec_j) if rec_j else np.nan)
        als_rec.append(np.mean(rec_a))
        ortho_rec.append(np.mean(rec_o))
        jenn_fac.append(np.mean(fac_j) if fac_j else np.nan)
        als_fac.append(np.mean(fac_a))
        ortho_fac.append(np.mean(fac_o))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(sigmas, jenn_rec, label='Jennrich Rec Err')
    ax1.plot(sigmas, als_rec, label='ALS Rec Err')
    ax1.plot(sigmas, ortho_rec, label='Ortho-ALS Rec Err')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_xlabel('sigma')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('Reconstruction Error vs sigma')

    ax2.plot(sigmas, jenn_fac, label='Jennrich Fac Sim')
    ax2.plot(sigmas, als_fac, label='ALS Fac Sim')
    ax2.plot(sigmas, ortho_fac, label='Ortho-ALS Fac Sim')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_xlabel('sigma')
    ax2.set_ylabel('Avg Cosine Similarity')
    ax2.set_title('Factor Similarity vs sigma')

    plt.tight_layout()
    plt.savefig('exp2.png')
    plt.show()

# Experiment 3: Initialization Sensitivity
def experiment3(d=20, k=3, sigma=0.1, rho=0.5, num_inits=20, num_reps=20):
    var_j_rec, var_a_rec, var_j_fac, var_a_fac = [], [], [], [] 
    var_o_rec, var_o_fac = [], []
    for _ in range(num_reps):
        T, T_noisy, A_true, B_true, C_true = generate_data_gmm(d, k, sigma, rho)
        # Jennrich
        rec_js, fac_js = [], []
        for _ in range(num_inits):
            A_hat, B_hat, C_hat = jennrich(T_noisy, k)
            if A_hat is not None:
                hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
                rec_js.append(reconstruction_error(T, hat_T))
                fac_js.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
        var_j_rec.append(np.std(rec_js) if rec_js else np.nan)
        var_j_fac.append(np.std(fac_js) if fac_js else np.nan)
        # ALS
        rec_as, fac_as = [], []
        for _ in range(num_inits):
            A_hat, B_hat, C_hat = cp_als(T_noisy, k)
            hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
            rec_as.append(reconstruction_error(T, hat_T))
            fac_as.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
        var_a_rec.append(np.std(rec_as))
        var_a_fac.append(np.std(fac_as))
        # Ortho-ALS
        rec_os, fac_os = [], []
        for _ in range(num_inits):
            A_hat, B_hat, C_hat = ortho_als(T_noisy, k)
            hat_T = reconstruct_tensor(A_hat, B_hat, C_hat)
            rec_os.append(reconstruction_error(T, hat_T))
            fac_os.append((factor_recovery_error(A_true, A_hat) + factor_recovery_error(B_true, B_hat) + factor_recovery_error(C_true, C_hat)) / 3)
        var_o_rec.append(np.std(rec_os))
    print('Mean STD Jennrich Rec Err:', np.nanmean(var_j_rec))
    print('Mean STD ALS Rec Err:', np.nanmean(var_a_rec))
    print('Mean STD Ortho-ALS Rec Err:', np.nanmean(var_o_rec))
    print('Mean STD Jennrich Fac Sim:', np.nanmean(var_j_fac))
    print('Mean STD ALS Fac Sim:', np.nanmean(var_a_fac))
    print('Mean STD Ortho-ALS Fac Sim:', np.nanmean(var_o_fac))