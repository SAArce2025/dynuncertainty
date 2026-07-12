import numpy as np
from scipy.integrate import solve_ivp


def simulate_single_trajectory(model, x0, params, t_span, t_eval, method="RK45"):
    """
    Simulate one trajectory of an ODE system.
    """
    solution = solve_ivp(
        fun=lambda t, x: model(t, x, params),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method=method,
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    return solution.y.T


def simulate_ensemble(model, initial_conditions, parameter_samples, t_span, t_eval, method="RK45"):
    """
    Simulate many trajectories.

    Returns:
        trajectories with shape:
        (n_samples, n_times, state_dimension)
    """
    if len(initial_conditions) != len(parameter_samples):
        raise ValueError("initial_conditions and parameter_samples must have the same length.")

    trajectories = []

    for x0, params in zip(initial_conditions, parameter_samples):
        trajectory = simulate_single_trajectory(
            model=model,
            x0=x0,
            params=params,
            t_span=t_span,
            t_eval=t_eval,
            method=method,
        )

        trajectories.append(trajectory)

    return np.array(trajectories)