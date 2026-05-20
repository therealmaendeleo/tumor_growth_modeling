import numpy as np


def tic_ode_system(
    t,
    y,
    a,
    b,
    c,
    mu,
    d,
    p,
    lmbda,
    eta_c_func,
    eta_mu_func,
    s_A_func,
    s_C_func,
):
    # При векторизации y имеет форму (3, n)
    T, I, C = y[0], y[1], y[2]

    # Замените это:
    # T = max(0.0, T)

    # На это:
    T = np.maximum(0.0, T)
    I = np.maximum(0.0, I)
    C = np.maximum(0.0, C)

    # Далее ваши уравнения...
    eta_c = eta_c_func(t)
    eta_mu = eta_mu_func(t)
    s_A = s_A_func(t)
    s_C = s_C_func(t)

    dTdt = a * T * (1 - b * T) - (c + eta_c) * T * I
    dIdt = (d + p * C) * T * I - (mu - eta_mu) * I + s_A
    dCdt = s_C - lmbda * C

    return np.array([dTdt, dIdt, dCdt])
