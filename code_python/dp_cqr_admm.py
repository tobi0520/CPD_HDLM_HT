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


def DP_regression(X, y, tau, gamma, delta):
    n = len(y)
    p = X.shape[1]
    b = 0
    bestvalue = np.zeros(n+1)
    partition = np.zeros(n+1)

    bestvalue[0] = -gamma

    for i in range(1, n+1):
        bestvalue[i] = float('inf')
        for l in range(1, i+1):
            b = bestvalue[l-1] + cqr_loss(X, y, l, i, tau, gamma)
            if b < bestvalue[i]:
                bestvalue[i] = b
                partition[i] = l-1

    R = n
    L = partition[R]

    while R > 0:
        R = int(L)
        L = partition[R]

    return partition[1:n+1].astype(int)


def part2local(parti_vec):
    N = len(parti_vec)
    localization = []
    r = N
    l = parti_vec[r - 1]
    localization.append(l)
    while l > 0:
        r = l
        l = parti_vec[r - 1]
        localization.insert(0, l)
    return localization[1:]
