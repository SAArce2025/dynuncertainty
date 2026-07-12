import numpy as np


def harmonic_oscillator(t, state, params):
    """
    Harmonic oscillator with uncertain mass and spring constant.

    State:
        state = [x, v]

    Parameters:
        params["m"] : mass
        params["k"] : spring constant
    """
    x, v = state

    m = params["m"]
    k = params["k"]

    dxdt = v
    dvdt = -(k / m) * x

    return np.array([dxdt, dvdt])