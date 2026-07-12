import numpy as np
import matplotlib.pyplot as plt

def plot_parameter_bounds(parameter_intervals, nominal_values=None, title="Parameter uncertainty bounds"):
    """
    Plot parameter uncertainty bounds.

    Parameters
    ----------
    parameter_intervals : dict
        Dictionary of parameter intervals.

        Example:
            {
                "m": (0.95, 1.05),
                "k": (0.95, 1.05),
                "x0": (0.95, 1.05),
            }

    nominal_values : dict, optional
        Dictionary of nominal parameter values.

        Example:
            {
                "m": 1.0,
                "k": 1.0,
                "x0": 1.0,
            }

    title : str
        Plot title.
    """
    names = list(parameter_intervals.keys())
    y_positions = np.arange(len(names))

    lows = np.array([parameter_intervals[name][0] for name in names])
    highs = np.array([parameter_intervals[name][1] for name in names])
    centers = 0.5 * (lows + highs)
    half_widths = 0.5 * (highs - lows)

    plt.figure(figsize=(8, 0.7 * len(names) + 2))

    plt.errorbar(
        centers,
        y_positions,
        xerr=half_widths,
        fmt="o",
        capsize=6,
        label="uncertainty interval",
    )

    if nominal_values is not None:
        nominal_x = [nominal_values.get(name, np.nan) for name in names]

        plt.scatter(
            nominal_x,
            y_positions,
            marker="x",
            s=80,
            label="nominal value",
        )

    for i, name in enumerate(names):
        low = lows[i]
        high = highs[i]

        plt.text(
            low,
            y_positions[i] + 0.18,
            f"{low:.4g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        plt.text(
            high,
            y_positions[i] + 0.18,
            f"{high:.4g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.yticks(y_positions, names)
    plt.xlabel("parameter value")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()