from cqr_admm import cqrpadmm
import numpy as np


def cqr_loss_from_beta(X, y, beta, alpha, tau):

    residual_sum = 0
    for k in range(len(tau)):
        residual = y - np.dot(X, beta) - alpha[k]
        residual_mat = np.zeros((len(y)))

        for i in range(residual.shape[0]):
            residual_mat[i] = max(tau[k] * residual[i],
                                  (tau[k] - 1) * residual[i])

        residual_sum += np.sum(residual_mat)

    return residual_sum


def cqr_loss(X, y, s, e, tau, gamma):
    p = X.shape[1]
    n = X.shape[0]
    if e - s <= 0:
        residual_s = 0
        alpha_temp = np.zeros(p)
        beta_cqr = None
    else:
        X_temp = X[s:e]
        y_temp = y[s:e]
        Xcenter = X_temp-X_temp.mean(axis=0)
        hdcqr = cqrpadmm(Xcenter, y_temp, intercept=False)
        cqradmm = hdcqr.cqrp_admm_smw(tau=tau)
        beta_cqr = cqradmm['beta']
        if max(beta_cqr) > 10e3:
            residual_s = 10e6
            beta_cqr = np.ones(p)*10
            alpha_temp = np.ones(len(tau))
        else:
            alpha_temp = cqradmm['alpha']
            residual_sum = cqr_loss_from_beta(
                X_temp, y_temp, beta_cqr, alpha_temp, tau)
            residual_s = residual_sum + gamma
    # return {'residual': residual_s}
    return {'residual': residual_s, 'beta': beta_cqr, 'alpha': alpha_temp}


def find_cp_one(X, y, s, e, tau, gamma, delta):

    if e - s <= 2 * delta:
        can_vec = [s]
    else:
        can_vec = [s] + list(range(s + delta, e - delta + 1))
    res_seq = [cqr_loss(X, y, s, t, tau, gamma)['residual'] +
               cqr_loss(X, y, t, e, tau, gamma)['residual']
               for t in can_vec]
    cp = can_vec[np.argmin(res_seq)]
    loss = np.min(res_seq)
    return {'cp': cp, 'loss': loss, 'res_seq': res_seq}


def binary_seg_cqr(X, y, tau, gamma, delta):
    n = X.shape[0]
    start_set = [0, n]
    output_set = []
    while len(start_set) > 1:

        s = start_set[0]

        e = start_set[1]
        # print(s, e)
        temp = find_cp_one(X, y, s, e, tau, gamma, delta)
        v = temp['cp']
        if v != start_set[0]:
            start_set = sorted(start_set + [v])
            output_set.append(v)
        else:
            start_set = start_set[1:]
    return sorted(output_set)
