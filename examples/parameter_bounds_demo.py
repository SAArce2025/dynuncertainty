import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from dynuncertainty.visualization import plot_parameter_bounds


def main():
    parameter_intervals = {
        "m": (0.95, 1.05),
        "k": (0.95, 1.05),
        "x0": (0.95, 1.05),
        "v0": (-0.02, 0.02),
    }

    nominal_values = {
        "m": 1.0,
        "k": 1.0,
        "x0": 1.0,
        "v0": 0.0,
    }

    plot_parameter_bounds(
        parameter_intervals=parameter_intervals,
        nominal_values=nominal_values,
        title="Harmonic oscillator: parameter bounds",
    )


if __name__ == "__main__":
    main()