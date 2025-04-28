import math, ROOT
from scipy.special import betainc
from scipy.stats import norm

def zbi(s, b, sigma_b_frac):
    if b <= 0:
        return 0.0
    sigma_b = sigma_b_frac * b
    tau = 1.0 / (b * sigma_b_frac * sigma_b_frac)
    n_on = s + b
    n_off = b * tau

    # probability
    P_Bi = betainc(n_on, n_off + 1, 1.0 / (1.0 + tau))

    if P_Bi <= 0:
        return 0.0

    # finally, ZBi is quantile (inverse CDF of normal) at 1 - P_Bi
    Z_Bi = norm.ppf(1.0 - P_Bi)

    return Z_Bi



print("ROOT output: ", ROOT.RooStats.NumberCountingUtils.BinomialExpZ(10, 100, 0.3))
print("My output: ", zbi(10, 100, 0.3))

