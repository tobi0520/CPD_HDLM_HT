import numpy as np
from sklearn.linear_model import Lasso

from ols import lasso_regression_seq





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



def DP_regression(X,y, gamma, lambda_, delta, tol=0.0001):

    n = len(y)
    p = X.shape[1]
    b = 0
    bestvalue = np.zeros(n+1)
    partition = np.zeros(n+1)

    bestvalue[0] = -gamma*np.log(max(n,p))

    for i in range(1, n+1):
        bestvalue[i] = float('inf')
        for l in range(1, i+1):
            b = bestvalue[l-1] + gamma*np.log(max(n,p)) + error_pred_seg_regression(X,y, l, i, lambda_, delta, tol)[0]
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