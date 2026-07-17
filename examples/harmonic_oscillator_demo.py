import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from dynuncertainty.models import harmonic_oscillator
from dynuncertainty.sampling import sample_initial_conditions, sample_parameter_intervals
from dynuncertainty.ensemble import simulate_ensemble, simulate_single_trajectory


def main():
    n_samples = 300

    t_span = (0.0, 30.0)
    t_eval = np.linspace(t_span[0], t_span[1], 800)

    initial_condition_intervals = [
        (0.95, 1.05),     # x0
        (-0.02, 0.02),    # v0
    ]

    parameter_intervals = {
        "m": (0.95, 1.05),
        "k": (0.95, 1.05),
    }

    initial_conditions = sample_initial_conditions(
        intervals=initial_condition_intervals,
        n_samples=n_samples,
        rng=123,
    )

    parameter_samples = sample_parameter_intervals(
        parameter_intervals=parameter_intervals,
        n_samples=n_samples,
        rng=456,
    )

    trajectories = simulate_ensemble(
        model=harmonic_oscillator,
        initial_conditions=initial_conditions,
        parameter_samples=parameter_samples,
        t_span=t_span,
        t_eval=t_eval,
    )

    nominal_params = {"m": 1.0, "k": 1.0}
    nominal_x0 = np.array([1.0, 0.0])

    nominal_trajectory = simulate_single_trajectory(
        model=harmonic_oscillator,
        x0=nominal_x0,
        params=nominal_params,
        t_span=t_span,
        t_eval=t_eval,
    )

    x_values = trajectories[:, :, 0]
    v_values = trajectories[:, :, 1]

    x_lower = np.percentile(x_values, 5, axis=0)
    x_median = np.percentile(x_values, 50, axis=0)
    x_upper = np.percentile(x_values, 95, axis=0)

    plt.figure()
    plt.fill_between(t_eval, x_lower, x_upper, alpha=0.3, label="5%-95% band")
    plt.plot(t_eval, x_median, label="median x(t)")
    plt.plot(t_eval, nominal_trajectory[:, 0], linestyle="--", label="nominal x(t)")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.title("Harmonic oscillator: uncertainty in x(t)")
    plt.legend()
    plt.grid(True)

    plt.figure()
    for trajectory in trajectories[::20]:
        plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.4)

    plt.plot(
        nominal_trajectory[:, 0],
        nominal_trajectory[:, 1],
        linewidth=2,
        label="nominal trajectory",
    )

    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Phase-space ensemble")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()