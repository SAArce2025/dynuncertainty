"""Interactive interface for running dynuncertainty models."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button, RadioButtons, TextBox

import numpy as np

from dynuncertainty.ensemble import (
    simulate_ensemble,
    simulate_single_trajectory,
)
from dynuncertainty.model_spec import ModelSpec
from dynuncertainty.phase_envelope import (
    compute_phase_polar_coordinates,
    compute_radial_phase_band,
    compute_time_resolved_phase_tube,
    normalize_phase_coordinates,
    polar_boundary_to_phase_coordinates,
)
from dynuncertainty.sampling import (
    sample_initial_conditions,
    sample_parameter_intervals,
)
from dynuncertainty.visualization import (
    plot_simulation_results,
)

class InteractiveModelRunner:
    """Dynamic graphical interface for one dynamical model."""

    def __init__(self, model: ModelSpec) -> None:
        self.model = model
        self.parameter_boxes: dict[str, dict[str, TextBox]] = {}
        self.initial_condition_boxes: dict[str, dict[str, TextBox]] = {}
        self.result_figures: list[Figure] = []
        self.tube_smoothness_selector: RadioButtons | None = None        

        self.figure = plt.figure(figsize=(11, 7))
        
        window_manager = self.figure.canvas.manager

        if window_manager is not None:
            window_manager.set_window_title(
              "dynuncertainty — Interactive model runner"
            )

        self._build_interface()

        self.figure.canvas.mpl_connect(
            "close_event",
            self._on_interface_closed,
        )

    def _build_interface(self) -> None:
        """Construct all visible interface elements."""

        self.figure.suptitle(
            f"dynuncertainty — {self.model.name}",
            fontsize=16,
            fontweight="bold",
        )

        self.figure.text(
            0.08,
            0.89,
            self.model.description,
            fontsize=10,
        )

        self.figure.text(
            0.08,
            0.82,
            "Parameter",
            fontsize=11,
            fontweight="bold",
        )

        self.figure.text(
            0.36,
            0.82,
            "Minimum",
            fontsize=11,
            fontweight="bold",
        )

        self.figure.text(
            0.54,
            0.82,
            "Nominal",
            fontsize=11,
            fontweight="bold",
        )

        self.figure.text(
            0.72,
            0.82,
            "Maximum",
            fontsize=11,
            fontweight="bold",
        )
    
    
        self._create_parameter_controls()
        self._create_initial_condition_controls()
        self._create_time_controls()
        self._create_run_button()

        self.status_text = self.figure.text(
            0.08,
            0.07,
            "Ready. Configure the intervals and press Run simulation.",
            fontsize=10,
            color="darkgreen",
        )

    def _create_parameter_controls(self) -> None:
        """Create one dynamic row for every model parameter."""

        starting_y = 0.74
        parameter_count = len(self.model.parameters)

        if parameter_count <= 2:
            row_spacing = 0.09
        else:
            row_spacing = 0.07

        for index, parameter in enumerate(self.model.parameters):
            y_position = starting_y - index * row_spacing

            parameter_label = parameter.label

            if parameter.unit:
                parameter_label += f" [{parameter.unit}]"

            self.figure.text(
                0.08,
                y_position + 0.015,
                parameter_label,
                fontsize=11,
            )

            lower_box = TextBox(
                ax=self.figure.add_axes(
                    (0.34, y_position, 0.13, 0.05)
                ),
                label="",
                initial=str(parameter.lower),
            )

            nominal_box = TextBox(
                ax=self.figure.add_axes(
                    (0.52, y_position, 0.13, 0.05)
                ),
                label="",
                initial=str(parameter.nominal),
            )

            upper_box = TextBox(
                ax=self.figure.add_axes(
                    (0.70, y_position, 0.13, 0.05)
                ),
                label="",
                initial=str(parameter.upper),
            )

            self.parameter_boxes[parameter.name] = {
                "lower": lower_box,
                "nominal": nominal_box,
                "upper": upper_box,
            }
            

    def _create_initial_condition_controls(self) -> None:
        """Create one dynamic row for every initial condition."""

        self.figure.text(
            0.08,
            0.55,
            "Initial conditions",
            fontsize=11,
            fontweight="bold",
        )

        starting_y = 0.47
        row_spacing = 0.09

        for index, condition in enumerate(
            self.model.initial_conditions
        ):
            y_position = starting_y - index * row_spacing

            condition_label = condition.label

            if condition.unit:
                condition_label += f" [{condition.unit}]"

            self.figure.text(
                0.08,
                y_position + 0.015,
                condition_label,
                fontsize=11,
            )

            lower_box = TextBox(
                ax=self.figure.add_axes(
                    (0.34, y_position, 0.13, 0.05)
                ),
                label="",
                initial=str(condition.lower),
            )

            nominal_box = TextBox(
                ax=self.figure.add_axes(
                    (0.52, y_position, 0.13, 0.05)
                ),
                label="",
                initial=str(condition.nominal),
            )

            upper_box = TextBox(
                ax=self.figure.add_axes(
                    (0.70, y_position, 0.13, 0.05)
                ),
                label="",
                initial=str(condition.upper),
            )

            self.initial_condition_boxes[condition.name] = {
                "lower": lower_box,
                "nominal": nominal_box,
                "upper": upper_box,
            }

    def _create_time_controls(self) -> None:
        """Create controls for the simulation time."""

        self.figure.text(
            0.08,
            0.28,
            "Simulation settings",
            fontsize=11,
            fontweight="bold",
        )

        self.time_start_box = TextBox(
            ax=self.figure.add_axes(
                (0.08, 0.20, 0.18, 0.05)
            ),
            label="Start time  ",
            initial=str(self.model.default_t_start),
        )

        self.time_end_box = TextBox(
            ax=self.figure.add_axes(
                (0.36, 0.20, 0.18, 0.05)
            ),
            label="End time  ",
            initial=str(self.model.default_t_end),
        )

        self.time_points_box = TextBox(
            ax=self.figure.add_axes(
                (0.64, 0.20, 0.18, 0.05)
            ),
            label="Points  ",
            initial=str(self.model.default_time_points),
        )

        self.samples_box = TextBox(
            ax=self.figure.add_axes(
                (0.08, 0.11, 0.18, 0.05)
            ),
            label="Samples  ",
            initial=str(self.model.default_n_samples),
        )

        self.seed_box = TextBox(
            ax=self.figure.add_axes(
                (0.36, 0.11, 0.18, 0.05)
            ),
            label="Seed  ",
            initial=str(self.model.default_seed),
        )

    def _create_run_button(self) -> None:
        """Create the simulation button."""

        button_axis = self.figure.add_axes(
            (0.64, 0.10, 0.24, 0.065)
        )

        self.run_button = Button(
            button_axis,
            "Run simulation",
            color="lightsteelblue",
            hovercolor="cornflowerblue",
        )

        self.run_button.on_clicked(self._on_run_clicked)

    def _read_configuration(self) -> dict:
        """Read and validate the values entered by the user."""

        parameter_intervals = {}
        nominal_parameters = {}
        initial_condition_intervals = {}
        nominal_initial_conditions = {}

        for parameter in self.model.parameters:
            boxes = self.parameter_boxes[parameter.name]

            lower = float(boxes["lower"].text)
            nominal = float(boxes["nominal"].text)
            upper = float(boxes["upper"].text)

            if lower > nominal or nominal > upper:
                raise ValueError(
                    f"{parameter.name}: values must satisfy "
                    "minimum <= nominal <= maximum."
                )

            parameter_intervals[parameter.name] = (
                lower,
                upper,
            )
            nominal_parameters[parameter.name] = nominal

        for condition in self.model.initial_conditions:
            boxes = self.initial_condition_boxes[condition.name]

            lower = float(boxes["lower"].text)
            nominal = float(boxes["nominal"].text)
            upper = float(boxes["upper"].text)

            if lower > nominal or nominal > upper:
                raise ValueError(
                    f"{condition.name}: values must satisfy "
                    "minimum <= nominal <= maximum."
                )

            initial_condition_intervals[condition.name] = (
                lower,
                upper,
            )
            nominal_initial_conditions[condition.name] = nominal

        start_time = float(self.time_start_box.text)
        end_time = float(self.time_end_box.text)
        time_points = int(self.time_points_box.text)
        n_samples = int(self.samples_box.text)
        seed = int(self.seed_box.text)


        if end_time <= start_time:
            raise ValueError(
                "End time must be greater than start time."
            )

        if time_points < 2:
            raise ValueError(
                "The number of time points must be at least 2."
            )

        if n_samples < 1:
            raise ValueError(
                "The number of samples must be at least 1."
            )

        if seed < 0:
            raise ValueError(
                "The random seed cannot be negative."
            )

        return {
            "model_key": self.model.key,
            "parameter_intervals": parameter_intervals,
            "nominal_parameters": nominal_parameters,
            "initial_condition_intervals": (
                initial_condition_intervals
            ),
            "nominal_initial_conditions": (
                nominal_initial_conditions
            ),
            "time_interval": (start_time, end_time),
            "time_points": time_points,
            "n_samples": n_samples,
            "seed": seed,
        }

    def _on_run_clicked(self, _event) -> None:
        """
        Read the configuration, run the ensemble simulation,
        and display the temporal and phase-space results.
        """

        try:
            configuration = self._read_configuration()
            
            start_time, end_time = configuration["time_interval"]

            t_eval = np.linspace(
                start_time,
                end_time,
                configuration["time_points"],
            )
           
            configuration_lines = []

            for parameter in self.model.parameters:
                lower, upper = configuration[
                    "parameter_intervals"
                ][parameter.name]

                nominal = configuration[
                    "nominal_parameters"
                ][parameter.name]

                if parameter.name.endswith("0"):
                    base_symbol = parameter.name[:-1]
                    symbol = (
                        rf"{base_symbol}_{{0,\mathrm{{nom}}}}"
                    )
                else:
                    symbol = (
                        rf"{parameter.name}_{{\mathrm{{nom}}}}"
                    )
                unit = f" {parameter.unit}" if parameter.unit else ""

                configuration_lines.append(
                    rf"${symbol}"
                    rf"={nominal:g}"
                    rf"\in[{lower:g},{upper:g}]$"
                    f"{unit}"
                )

                
            for condition in self.model.initial_conditions:
                lower, upper = configuration[
                    "initial_condition_intervals"
                ][condition.name]

                nominal = configuration[
                    "nominal_initial_conditions"
                ][condition.name]

                if condition.name.endswith("0"):
                    base_symbol = condition.name[:-1]

                    if base_symbol == "theta":
                        displayed_symbol = r"\theta"
                    elif base_symbol == "omega":
                        displayed_symbol = r"\omega"
                    else:
                        displayed_symbol = base_symbol

                    symbol = (
                        rf"{displayed_symbol}_{{0,\mathrm{{nom}}}}"
                    )
                else:
                    symbol = (
                        rf"{condition.name}_{{\mathrm{{nom}}}}"
                    )

                unit = (
                    f" {condition.unit}"
                    if condition.unit
                    else ""
                )

                configuration_lines.append(
                    rf"${symbol}"
                    rf"={nominal:g}"
                    rf"\in[{lower:g},{upper:g}]$"
                    f"{unit}"
                )

            configuration_lines.append(
                f"Monte Carlo samples: N = "
                f"{configuration['n_samples']}"
            )

            configuration_lines.append(
                f"Random seed: {configuration['seed']}"
            )

            configuration_text = "\n".join(
                configuration_lines
            )

            configuration_text = "\n".join(configuration_lines)

            n_samples = configuration["n_samples"]
            rng = np.random.default_rng(
                configuration["seed"]
            )


            initial_intervals = [
                configuration["initial_condition_intervals"][
                    condition.name
                ]
                for condition in self.model.initial_conditions
            ]

            initial_samples = sample_initial_conditions(
                intervals=initial_intervals,
                n_samples=n_samples,
                rng=rng,
            )

            parameter_samples = sample_parameter_intervals(
                parameter_intervals=configuration[
                    "parameter_intervals"
                ],
                n_samples=n_samples,
                rng=rng,
            )

            trajectories = simulate_ensemble(
                model=self.model.derivative,
                initial_conditions=initial_samples,
                parameter_samples=parameter_samples,
                t_span=(start_time, end_time),
                t_eval=t_eval,
            )
            
            nominal_initial_state = np.array(
                [
                    configuration["nominal_initial_conditions"][
                        condition.name
                    ]
                    for condition in self.model.initial_conditions
                ],
                dtype=float,
            )

            nominal_trajectory = simulate_single_trajectory(
                model=self.model.derivative,
                x0=nominal_initial_state,
                params=configuration["nominal_parameters"],
                t_span=(start_time, end_time),
                t_eval=t_eval,
            )

            nominal_position = nominal_trajectory[:, 0]

            positions = trajectories[:, :, 0]
            velocities = trajectories[:, :, 1]

            if self.model.phase_reference_frequency is None:
                raise ValueError(
                    "The selected model does not define a "
                    "phase-space reference frequency."
                )

            nominal_frequency = (
                self.model.phase_reference_frequency(
                    configuration["nominal_parameters"]
                )
            )

            if nominal_frequency <= 0:
                raise ValueError(
                    "The phase-space reference frequency "
                    "must be positive."
                )

            normalized_positions, normalized_velocities = (
                normalize_phase_coordinates(
                    positions=positions,
                    velocities=velocities,
                    reference_frequency=nominal_frequency,
                )
            )

            inner_band_positions = None
            inner_band_velocities = None
            outer_band_positions = None
            outer_band_velocities = None

            tube_centre_positions = None
            tube_centre_velocities = None
            tube_radii = None

            if self.model.phase_band_strategy == "radial":
                phase_radii, phase_angles = (
                    compute_phase_polar_coordinates(
                        normalized_positions=normalized_positions,
                        normalized_velocities=normalized_velocities,
                    )
                )

                (
                    band_angles,
                    inner_band_radii,
                    outer_band_radii,
                ) = compute_radial_phase_band(
                    radii=phase_radii,
                    angles=phase_angles,
                    n_angular_bins=360,
                )

                (
                    inner_band_positions,
                    inner_band_velocities,
                ) = polar_boundary_to_phase_coordinates(
                    angles=band_angles,
                    radii=inner_band_radii,
                    reference_frequency=nominal_frequency,
                )

                (
                    outer_band_positions,
                    outer_band_velocities,
                ) = polar_boundary_to_phase_coordinates(
                    angles=band_angles,
                    radii=outer_band_radii,
                    reference_frequency=nominal_frequency,
                )

            elif self.model.phase_band_strategy == "trajectory_tube":
                (
                    tube_centre_positions,
                    tube_centre_velocities,
                    tube_radii,
                ) = compute_time_resolved_phase_tube(
                    normalized_positions=normalized_positions,
                    normalized_velocities=normalized_velocities,
                    coverage_percentile=90.0,
                    n_time_slices=60,
                )

            else:
                raise ValueError(
                    "Unknown phase-band strategy: "
                    f"{self.model.phase_band_strategy!r}."
                )

            lower_position = np.percentile(
                positions,
                5,
                axis=0,
            )

            median_position = np.median(
                positions,
                axis=0,
            )

            upper_position = np.percentile(
                positions,
                95,
                axis=0,
            )

            
            results_figure = plot_simulation_results(
                times=t_eval,
                lower_position=lower_position,
                median_position=median_position,
                upper_position=upper_position,
                nominal_position=nominal_position,
                positions=positions,
                velocities=velocities,
                nominal_trajectory=nominal_trajectory,
                inner_band_positions=inner_band_positions,
                inner_band_velocities=inner_band_velocities,
                outer_band_positions=outer_band_positions,
                outer_band_velocities=outer_band_velocities,
                model_name=self.model.name,
                configuration_text=configuration_text,
                n_samples=n_samples,
                state_label=self.model.state_labels[0],
                phase_state_labels=self.model.state_labels,
                phase_band_strategy=(
                    self.model.phase_band_strategy
                ),
                tube_centre_positions=tube_centre_positions,
                tube_centre_velocities=tube_centre_velocities,
                tube_radii=tube_radii,
                reference_frequency=nominal_frequency,
            )
            
            self.result_figures.append(results_figure)
            
            results_figure.show()

        except ValueError as error:
            self.status_text.set_text(f"Error: {error}")
            self.status_text.set_color("crimson")
            self.figure.canvas.draw_idle()
            return

        self.status_text.set_text(
            "Simulation completed successfully."
        )
        self.status_text.set_color("darkgreen")

        print("\nAccepted simulation configuration:")
        print(configuration)

        self.figure.canvas.draw_idle()

    def _on_interface_closed(self, _event) -> None:
        """Close every result figure created by this interface."""

        for result_figure in self.result_figures:
            plt.close(result_figure)

        self.result_figures.clear()

    def show(self) -> None:
        """Display the interface."""

        plt.show()