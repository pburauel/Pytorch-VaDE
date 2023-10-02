import torch
import torch.nn.functional as F

import numpy as np
from scipy.stats import gamma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rbf_dot_torch(pattern1, pattern2, deg):
    size1 = pattern1.size()
    size2 = pattern2.size()

    G = torch.sum(pattern1 * pattern1, dim=1).reshape(size1[0], 1)
    H = torch.sum(pattern2 * pattern2, dim=1).reshape(size2[0], 1)

    Q = G.repeat(1, size2[0])
    R = H.T.repeat(size1[0], 1)

    H = Q + R - 2 * torch.mm(pattern1, pattern2.T)

    H = torch.exp(-H / (2 * deg ** 2))

    return H

def hsic_gam_torch(X, Y, alph=0.05):
    """
    X, Y are torch tensors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width
    """
    n = X.size(0)

    # ----- width of X -----
    Xmed = X

    G = torch.sum(Xmed * Xmed, dim=1).reshape(n, 1)
    Q = G.repeat(1, n)
    R = G.T.repeat(n, 1)

    dists = Q + R - 2 * torch.mm(Xmed, Xmed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_x = torch.sqrt(0.5 * torch.median(dists[dists > 0]))
    # ----- -----

    # ----- width of X -----
    Ymed = Y

    G = torch.sum(Ymed * Ymed, dim=1).reshape(n, 1)
    Q = G.repeat(1, n)
    R = G.T.repeat(n, 1)

    dists = Q + R - 2 * torch.mm(Ymed, Ymed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_y = torch.sqrt(0.5 * torch.median(dists[dists > 0]))
    # ----- -----

    bone = torch.ones((n, 1), dtype=torch.float32)
    H = torch.eye(n) - torch.ones((n, n), dtype=torch.float32) / n

    K = rbf_dot_torch(X, X, width_x)
    L = rbf_dot_torch(Y, Y, width_y)
    
    # print(f'H device is {H.get_device()}')
    # print(f'K device is {K.get_device()}')
    # print(f'K device is {K.to(device).get_device()}')
    # print(f'H device is {H.get_device()}')

    # Kc = torch.mm(torch.mm(H, K), H).cuda()
    Kc = torch.mm(torch.mm(H.to(device), K.to(device)), H.to(device)).to(device)
    # Lc = torch.mm(torch.mm(H, L), H).cuda()
    Lc = torch.mm(torch.mm(H.to(device), L.to(device)), H.to(device)).to(device)


    testStat = torch.sum(Kc.T * Lc) / n

    
    varHSIC = (Kc * Lc / 6)**2
    varHSIC = ( torch.sum(varHSIC) - torch.trace(varHSIC) ) / n / (n-1)
    varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)
    K = K - torch.diag(torch.diag(K))
    L = L - torch.diag(torch.diag(L))
    muX = torch.mm(torch.mm(bone.T, K), bone) / n / (n-1)
    muY = torch.mm(torch.mm(bone.T, L), bone) / n / (n-1)
    mHSIC = (1 + muX * muY - muX - muY) / n
    al = mHSIC**2 / varHSIC
    bet = varHSIC*n / mHSIC
    
    # For operations not available in PyTorch, you can detach the tensor and convert to numpy
    al_numpy = al.cpu().detach().numpy()
    bet_numpy = bet.cpu().detach().numpy()
    thresh = gamma.ppf(1-alph, al_numpy, scale=bet_numpy)[0][0]
    return (testStat.numpy(), thresh)