import numpy as np
from sklearn.linear_model import Lasso

from ols import lasso_regression_seq
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)



def CQRPCDCPP(X, y, beta, beta0, toler, maxit, tau, lamda):
    n = X.shape[0]  
    k = tau.shape[0]  
    p = X.shape[1]  
    error = 10000
    iteration = 1
    u = np.zeros(k)
    r = np.zeros((n, k))
    signw = np.zeros((n, k))
    z = np.zeros((n, k))
    newX = np.zeros((n, k))

    while iteration <= maxit and error > toler:
        betaold = beta.copy()
        uv = np.sort(y - np.dot(X, beta))

        quantile = (n - 1) * tau - np.floor((n - 1) * tau)

        for i in range(k):
            u[i] = quantile[i] * uv[int(np.ceil((n - 1) * tau[i]))] + (
                1 - quantile[i]) * uv[int(np.floor((n - 1) * tau[i]))]

        yh = np.dot(X, beta)

        for i in range(k):
            r[:, i] = y - u[i] - yh
            signw[:, i] = (1 - np.sign(r[:, i])) / 2 * \
                (1 - tau[i]) + np.sign(r[:, i]) * tau[i] / 2

        for j in range(p):
            xbeta = beta[j] * X[:, j]
            for i in range(k):
                z[:, i] = (r[:, i] + xbeta) / X[:, j]
                newX[:, i] = X[:, j] * signw[:, i]

            vz = z.flatten()
            vz = np.append(vz, 0)
            order = np.argsort(vz)
            sortz = vz[order]
            vnewX = newX.flatten()
            vnewX = np.append(vnewX, 0)
            vnewX[n * k] = lamda / (beta0[j] * beta0[j])
            w = np.abs(vnewX[order])
            cumsum_w = np.cumsum(w)
            index = np.argmax(cumsum_w > np.sum(w) / 2)
            place = int(index)

            beta[j] = sortz[place]

        error = np.sum(np.abs(beta - betaold))
        iteration += 1

    return beta


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


# def cqr_loss(X, y, s, e, tau, lambda_, gamma):
#     p = X.shape[1]
#     n = X.shape[0]
#     if e - s <= 0:
#         residual_s = 0
#         alpha_temp = np.zeros(p)
#         beta_cqr = None
#     else:
#         X_temp = X[s:e]
#         y_temp = y[s:e]
#         betaold = lasso_regression_seq(
#             X_temp, y_temp, lambda_ * np.sqrt(max(np.log(max(n, p)), (e-s))) / (e-s), 0.0001)

#         beta_cqr = CQRPCDCPP(X_temp, y_temp, betaold, np.ones(
#             p), 1e-3, 200, tau, lambda_ * np.sqrt(max(np.log(max(n, p)), (e-s))))
#         alpha_temp = np.percentile(
#             y_temp - np.dot(X_temp, beta_cqr), tau * 100)
#         residual_sum = cqr_loss_from_beta(
#             X_temp, y_temp, beta_cqr, alpha_temp, tau)
#         residual_s = residual_sum + gamma
#     # return {'residual': residual_s}
#     return {'residual': residual_s, 'beta': beta_cqr, 'alpha': alpha_temp}


def cqr_loss(X, y, s, e, tau, lambda_, gamma):
    p = X.shape[1]
    n = X.shape[0]
    if e - s <= 0:
        residual_s = 0
        alpha_temp = np.zeros(p)
        beta_cqr = None
    else:
        X_temp = X[s:e]
        y_temp = y[s:e]
        betaold = lasso_regression_seq(
            X_temp, y_temp,  lambda_ * np.sqrt(max(np.log(max(n, p)), (e-s))) / (e-s), 0.0001)

        beta_cqr = CQRPCDCPP(X_temp, y_temp, betaold, np.ones(
            p), 1e-3, 200, tau, lambda_ * np.sqrt(max(np.log(max(n, p)), (e-s))))
        if max(beta_cqr) > 10e2:
            residual_s = 10e6
            beta_cqr = np.ones(p)*10
            alpha_temp = np.ones(len(tau))
        else:
            alpha_temp = np.percentile(
                y_temp - np.dot(X_temp, beta_cqr), tau * 100)
            residual_sum = cqr_loss_from_beta(
                X_temp, y_temp, beta_cqr, alpha_temp, tau)
            residual_s = residual_sum + gamma
    # return {'residual': residual_s}
    return {'residual': residual_s, 'beta': beta_cqr, 'alpha': alpha_temp}


def find_cp_one(X, y, s, e, tau, lambda_, gamma, delta):

    if e - s <= 2 * delta:
        can_vec = [s]
    else:
        can_vec = [s] + list(range(s + delta, e - delta + 1))
    res_seq = [cqr_loss(X, y, s, t, tau, lambda_, gamma)['residual'] +
               cqr_loss(X, y, t, e, tau, lambda_, gamma)['residual']
               for t in can_vec]
    cp = can_vec[np.argmin(res_seq)]
    loss = np.min(res_seq)
    return {'cp': cp, 'loss': loss, 'res_seq': res_seq}


def binary_seg_cqr(X, y, tau, lambda_, gamma, delta):
    n = X.shape[0]
    start_set = [0, n]
    output_set = []
    while len(start_set) > 1:

        s = start_set[0]

        e = start_set[1]
        # print(s, e)
        temp = find_cp_one(X, y, s, e, tau, lambda_, gamma, delta)
        v = temp['cp']
        if v != start_set[0]:
            start_set = sorted(start_set + [v])
            output_set.append(v)
        else:
            start_set = start_set[1:]
    return sorted(output_set)
