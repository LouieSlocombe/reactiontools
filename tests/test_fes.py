"""Tests for reactiontools.tools_fes.

These only need numpy, pandas and matplotlib, so they run without ASE, a
calculator or an MD engine. conftest selects the Agg backend, so nothing tries
to open a window on a headless machine.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from reactiontools.tools_fes import (
    FES,
    FESSummary,
    as_fes,
    convert_energy,
    fes_convergence,
    fes_series_files,
    load_fes_series,
    plot_fes,
    plot_fes_1d,
    plot_fes_2d,
    plot_fes_2d_overlay,
    plot_fes_convergence,
    plot_fes_slices,
    plot_plumed_colvar,
    plot_plumed_fes,
    read_plumed_file,
    summarise_fes,
    unit_label,
)

EV_IN_KJ_PER_MOL = 96.48533212331


def write_fes_1d(path, n_bins=50, derivatives=True, infinities=False):
    """Write a 1-D PLUMED FES file and return the values it holds."""
    x = np.linspace(-np.pi, np.pi, n_bins)
    free = 20.0 * (1.0 - np.cos(2.0 * x)) + 5.0
    if infinities:
        free[:3] = np.inf

    with open(path, "w") as handle:
        handle.write(
            "#! FIELDS pt_cv file.free" + (" der_pt_cv\n" if derivatives else "\n")
        )
        handle.write(f"#! SET min_pt_cv -3.141593\n#! SET nbins_pt_cv {n_bins}\n")
        handle.write("#! SET periodic_pt_cv false\n")
        for cv, value in zip(x, free, strict=True):
            handle.write(f"{cv:.9f}  {value:.9f}" + (" 0.0\n" if derivatives else "\n"))
    return x, free


def write_fes_2d(path, n_x=21, n_y=17, derivatives=True, blocks=True):
    """Write a 2-D PLUMED FES file, x running fastest as PLUMED does."""
    grid_x, grid_y = np.meshgrid(
        np.linspace(-2.0, 2.0, n_x), np.linspace(0.0, 3.0, n_y)
    )
    free = 10.0 * (grid_x**2 - 1.0) ** 2 + 5.0 * (grid_y - 1.5) ** 2

    with open(path, "w") as handle:
        handle.write(
            "#! FIELDS cv_diff1 cv_diff2 file.free"
            + (" der_cv_diff1 der_cv_diff2\n" if derivatives else "\n")
        )
        handle.write("#! SET min_cv_diff1 -2\n")
        for i in range(n_y):
            for j in range(n_x):
                handle.write(
                    f"{grid_x[i, j]:.9f} {grid_y[i, j]:.9f}  {free[i, j]:.9f}"
                    + (" 0.0 0.0\n" if derivatives else "\n")
                )
            if blocks:
                handle.write("\n")  # PLUMED separates grid rows with a blank line
    return grid_x, grid_y, free


def write_colvar(path, n_rows=100):
    """Write a small PLUMED COLVAR file."""
    with open(path, "w") as handle:
        handle.write("#! FIELDS time pt_cv metad.bias\n")
        for i in range(n_rows):
            time = i * 0.002
            handle.write(f"{time:.4f} {np.sin(time):.6f} {0.1 * i:.6f}\n")


@pytest.fixture
def fes_1d_file(tmp_path):
    """Path to a 1-D PLUMED FES file."""
    path = tmp_path / "fes.dat"
    write_fes_1d(path)
    return str(path)


@pytest.fixture
def fes_2d_file(tmp_path):
    """Path to a 2-D PLUMED FES file."""
    path = tmp_path / "fes2d.dat"
    write_fes_2d(path)
    return str(path)


@pytest.fixture
def colvar_file(tmp_path):
    """Path to a PLUMED COLVAR file."""
    path = tmp_path / "COLVAR"
    write_colvar(path)
    return str(path)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def test_read_plumed_file_header_and_metadata(fes_1d_file):
    plumed = read_plumed_file(fes_1d_file)
    assert plumed.fields == ["pt_cv", "file.free"]  # der_* dropped
    assert plumed.data.shape == (50, 2)
    assert plumed.metadata["nbins_pt_cv"] == "50"
    assert plumed.metadata["periodic_pt_cv"] == "false"
    assert plumed.column("file.free").shape == (50,)
    assert list(plumed.to_dataframe().columns) == ["pt_cv", "file.free"]


def test_read_plumed_file_keeps_derivatives_when_asked(fes_1d_file):
    assert read_plumed_file(fes_1d_file, drop_der=False).data.shape == (50, 3)


def test_read_plumed_file_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.dat"
    path.write_text("#! FIELDS a b\n")
    with pytest.raises(ValueError, match="No numeric data"):
        read_plumed_file(str(path))


# ---------------------------------------------------------------------------
# Loading and preparing
# ---------------------------------------------------------------------------
def test_as_fes_1d(fes_1d_file):
    fes = as_fes(fes_1d_file)
    assert fes.ndim == 1
    assert fes.cv_labels == ["pt_cv"]
    assert np.isclose(np.nanmin(fes.energy), 0.0)  # shifted to zero
    assert fes.energy_unit == "kJ/mol"


def test_as_fes_2d_grid_round_trips(fes_2d_file):
    grid_x, grid_y, free = write_fes_2d(fes_2d_file)
    fes = as_fes(fes_2d_file)
    assert fes.ndim == 2 and fes.regular
    assert fes.energy.shape == (17, 21)
    assert np.allclose(fes.cvs[0], grid_x)
    assert np.allclose(fes.cvs[1], grid_y)
    assert np.allclose(fes.energy, free - free.min())


def test_as_fes_handles_files_without_block_separators(tmp_path, fes_2d_file):
    flat = tmp_path / "flat.dat"
    write_fes_2d(flat, blocks=False)
    assert np.allclose(as_fes(str(flat)).energy, as_fes(fes_2d_file).energy)


def test_as_fes_converts_units(fes_1d_file):
    in_kj = as_fes(fes_1d_file)
    in_ev = as_fes(fes_1d_file, energy_unit="eV")
    assert np.allclose(in_ev.energy * EV_IN_KJ_PER_MOL, in_kj.energy)
    assert in_ev.label == r"$F$ (eV)"


def test_as_fes_does_not_convert_twice(fes_1d_file):
    in_ev = as_fes(fes_1d_file, energy_unit="eV")
    assert np.allclose(as_fes(in_ev, energy_unit="eV").energy, in_ev.energy)
    assert np.allclose(as_fes(in_ev).energy, in_ev.energy)


def test_as_fes_ignores_infinities_when_shifting(tmp_path):
    path = tmp_path / "inf.dat"
    write_fes_1d(path, infinities=True)
    fes = as_fes(str(path))
    assert np.isfinite(fes.energy).sum() == 47
    assert np.isclose(np.nanmin(fes.energy), 0.0)


def test_as_fes_masks_above_max_energy(fes_2d_file):
    fes = as_fes(fes_2d_file, max_energy=5.0)
    assert np.nanmax(fes.energy) <= 5.0
    assert np.isnan(fes.energy).any()


def test_as_fes_does_not_mutate_its_input(fes_2d_file):
    original = as_fes(fes_2d_file)
    before = original.energy.copy()
    as_fes(original, max_energy=1.0, energy_unit="eV")
    assert np.allclose(original.energy, before, equal_nan=True)


def test_as_fes_rejects_single_column_files(tmp_path):
    path = tmp_path / "one.dat"
    path.write_text("#! FIELDS a\n1.0\n2.0\n")
    with pytest.raises(ValueError, match="at least 2 columns"):
        as_fes(str(path))


# ---------------------------------------------------------------------------
# Array inputs
# ---------------------------------------------------------------------------
def test_as_fes_accepts_every_array_layout(fes_2d_file):
    grid_x, grid_y, free = write_fes_2d(fes_2d_file)
    reference = as_fes(np.stack([grid_x, grid_y, free]))
    assert reference.ndim == 2 and reference.regular

    as_tuple = as_fes((grid_x, grid_y, free))
    as_columns = as_fes(np.column_stack([grid_x.ravel(), grid_y.ravel(), free.ravel()]))
    as_scattered = as_fes((grid_x.ravel(), grid_y.ravel(), free.ravel()))
    for other in (as_tuple, as_columns, as_scattered):
        assert np.allclose(other.energy, reference.energy)

    x = np.linspace(-1.0, 1.0, 30)
    free_1d = x**2
    rows = as_fes(np.vstack([x, free_1d]))  # (2, N), the package convention
    columns = as_fes(np.vstack([x, free_1d]).T)  # (N, 2)
    assert rows.ndim == 1
    assert np.allclose(rows.energy, columns.energy)


def test_as_fes_falls_back_to_triangulation_for_irregular_data():
    points = np.random.default_rng(0).random((97, 3))
    assert as_fes(points).regular is False


def test_as_fes_rejects_uninterpretable_arrays():
    with pytest.raises(ValueError, match="Cannot interpret"):
        as_fes(np.zeros((5, 7)))


def test_lists_of_surfaces_are_not_mistaken_for_one_surface(fes_2d_file):
    """``[a, b, c]`` is three curves, ``(X, Y, Z)`` is one surface."""
    grid_x, grid_y, free = write_fes_2d(fes_2d_file)
    curve = np.vstack([np.linspace(0, 1, 20), np.linspace(0, 1, 20) ** 2])

    _, ax = plot_fes_1d([curve, curve, curve])
    assert len(ax.lines) == 3

    _, ax = plot_fes_2d((grid_x, grid_y, free))
    assert np.atleast_1d(ax).size == 1


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
def test_convert_energy():
    assert np.isclose(convert_energy(EV_IN_KJ_PER_MOL, "kJ/mol", "eV"), 1.0)
    assert np.isclose(convert_energy(1.0, "kcal/mol", "kJ/mol"), 4.184)
    assert np.isclose(convert_energy(5.0, "eV", "eV"), 5.0)
    assert np.allclose(convert_energy([1.0, 2.0], "kJ/mol"), [1.0, 2.0])  # no target
    assert unit_label("EV") == r"$F$ (eV)"  # case insensitive
    with pytest.raises(KeyError):
        convert_energy(1.0, "kJ/mol", "furlongs")


# ---------------------------------------------------------------------------
# 1-D plotting
# ---------------------------------------------------------------------------
def test_plot_fes_1d_single_curve_has_no_legend(fes_1d_file):
    _, ax = plot_fes_1d(fes_1d_file)
    assert ax.get_legend() is None
    assert ax.get_xlabel() == "pt_cv"
    assert "kJ" in ax.get_ylabel()


def test_plot_fes_1d_series_labels_from_times(fes_1d_file):
    _, ax = plot_fes_1d(
        [fes_1d_file] * 3,
        labels=[10, 20, 30],
        label_template=r"$t={:g}$ ps",
        energy_unit="eV",
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        r"$t=10$ ps",
        r"$t=20$ ps",
        r"$t=30$ ps",
    ]
    assert ax.get_ylabel() == r"$F$ (eV)"


def test_plot_fes_1d_max_datasets_keeps_the_last_ones(fes_1d_file):
    _, ax = plot_fes_1d(
        [fes_1d_file] * 5, labels=list(range(5)), label_template="{:g}", max_datasets=2
    )
    assert len(ax.lines) == 2
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["3", "4"]


def test_plot_fes_1d_rejects_2d_input(fes_2d_file):
    with pytest.raises(ValueError, match="1-D free-energy surfaces"):
        plot_fes_1d(fes_2d_file)


def test_plot_fes_1d_checks_label_count(fes_1d_file):
    with pytest.raises(ValueError, match="labels for"):
        plot_fes_1d(fes_1d_file, labels=["a", "b"])


# ---------------------------------------------------------------------------
# 2-D plotting
# ---------------------------------------------------------------------------
def test_plot_fes_2d_single_panel_with_colorbar(fes_2d_file):
    fig, axes = plot_fes_2d(fes_2d_file)
    assert axes.size == 1
    assert len(fig.axes) == 2  # panel + colour bar
    assert axes[0].get_xlabel() == "cv_diff1"
    assert axes[0].get_ylabel() == "cv_diff2"


def test_plot_fes_2d_shares_one_colorbar_across_panels(fes_2d_file):
    fig, axes = plot_fes_2d(
        [fes_2d_file] * 3, labels=[1.0, 2.0, 3.0], label_template=r"$t={:g}$ ps"
    )
    assert axes.size == 3
    assert len(fig.axes) == 4  # three panels + one shared colour bar
    assert axes[1].get_title() == r"$t=2$ ps"
    assert axes[1].get_ylabel() == ""  # only the left-most panel is labelled


def test_plot_fes_2d_rejects_1d_input(fes_1d_file):
    with pytest.raises(ValueError, match="2-D free-energy surfaces"):
        plot_fes_2d(fes_1d_file)


def test_plot_fes_2d_checks_axes_count(fes_2d_file):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="axes for"):
        plot_fes_2d([fes_2d_file] * 3, fig=fig, ax=ax)


def test_plot_fes_2d_overlay_labels_each_surface(fes_2d_file):
    _, ax = plot_fes_2d_overlay(
        [fes_2d_file, fes_2d_file], labels=["MD", "PIMD"], levels=5
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["MD", "PIMD"]


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------
def test_slice_at_picks_the_right_row_and_column(fes_2d_file):
    grid_x, grid_y, free = write_fes_2d(fes_2d_file)
    fes = as_fes(fes_2d_file)
    shifted = free - free.min()

    # Hold CV1 fixed: the cut runs along CV2.
    cv, energy, used = fes.slice_at(-1.0, axis=0)
    assert np.isclose(used, -1.0)
    assert np.allclose(cv, grid_y[:, 5])
    assert np.allclose(energy, shifted[:, 5])

    # Hold CV2 fixed: the cut runs along CV1.
    cv, energy, used = fes.slice_at(1.5, axis=1)
    assert np.isclose(used, 1.5)
    assert np.allclose(cv, grid_x[8, :])
    assert np.allclose(energy, shifted[8, :])


def test_plot_fes_slices_draws_one_line_per_surface_and_value(fes_2d_file):
    _, ax = plot_fes_slices(
        [fes_2d_file, fes_2d_file],
        at=[-1.0, 0.0],
        labels=["MD", "PIMD"],
        energy_unit="eV",
    )
    assert len(ax.lines) == 4
    texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert texts[0] == r"MD, cv_diff1$=-1.00$"
    assert ax.get_xlabel() == "cv_diff2"


def test_slice_at_rejects_1d_surfaces(fes_1d_file):
    with pytest.raises(ValueError, match="2-D free-energy surface"):
        as_fes(fes_1d_file).slice_at(0.0)


# ---------------------------------------------------------------------------
# Dispatch and saving
# ---------------------------------------------------------------------------
def test_plot_fes_dispatches_on_dimensionality(fes_1d_file, fes_2d_file):
    _, ax = plot_fes(fes_1d_file, levels=10)  # 2-D options are ignored, not fatal
    assert isinstance(ax, plt.Axes)
    _, axes = plot_fes(fes_2d_file)
    assert np.atleast_1d(axes).size == 1


def test_plot_plumed_fes_returns_a_single_axes(fes_1d_file, fes_2d_file):
    _, ax = plot_plumed_fes(fes_1d_file)
    assert isinstance(ax, plt.Axes)
    _, ax = plot_plumed_fes(fes_2d_file, levels=12, energy_unit="eV")
    assert isinstance(ax, plt.Axes)


def test_plot_plumed_fes_draws_on_supplied_axes(fes_1d_file):
    fig, ax = plt.subplots()
    out_fig, out_ax = plot_plumed_fes(fes_1d_file, ax=ax)
    assert out_fig is fig and out_ax is ax


def test_saving_writes_only_what_was_asked_for(tmp_path, fes_1d_file):
    plot_fes_1d(fes_1d_file, filename=str(tmp_path / "both"))
    assert (tmp_path / "both.png").exists() and (tmp_path / "both.pdf").exists()

    plot_fes_1d(fes_1d_file, filename=str(tmp_path / "png_only.png"))
    assert (tmp_path / "png_only.png").exists()
    assert not (tmp_path / "png_only.pdf").exists()

    plot_fes_1d(fes_1d_file)  # nothing is written without a filename
    assert sorted(p.name for p in tmp_path.glob("*.p*")) == [
        "both.pdf",
        "both.png",
        "png_only.png",
    ]


def test_saving_picks_the_right_figure(tmp_path, fes_2d_file):
    """An unrelated open figure must not be the one that gets written."""
    other_fig, other_ax = plt.subplots()
    other_ax.plot([0, 1], [0, 1])
    fig, _ = plot_fes_2d(fes_2d_file, filename=str(tmp_path / "fes.png"))
    assert (tmp_path / "fes.png").exists()
    assert fig is not other_fig


# ---------------------------------------------------------------------------
# COLVAR
# ---------------------------------------------------------------------------
def test_plot_plumed_colvar_one_panel_per_variable(colvar_file):
    _, axes = plot_plumed_colvar(colvar_file)
    assert axes.size == 2  # pt_cv and metad.bias; time is the x-axis
    assert axes[0].get_ylabel() == "pt_cv"
    assert axes[-1].get_xlabel() == "time"


def test_plot_plumed_colvar_can_select_columns(colvar_file):
    _, axes = plot_plumed_colvar(colvar_file, columns=["pt_cv"])
    assert axes.size == 1
    with pytest.raises(ValueError, match="not found"):
        plot_plumed_colvar(colvar_file, columns=["nope"])


def test_plot_plumed_colvar_falls_back_to_the_index(colvar_file):
    _, axes = plot_plumed_colvar(colvar_file, x_axis="not_a_column")
    assert axes[-1].get_xlabel() == "Step (index)"


def test_plot_plumed_colvar_needs_a_header(tmp_path):
    path = tmp_path / "bare.dat"
    path.write_text("0.0 1.0\n1.0 2.0\n")
    with pytest.raises(ValueError, match="FIELDS"):
        plot_plumed_colvar(str(path))


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
def test_fes_defaults_and_range():
    fes = FES(cvs=[np.linspace(0, 1, 5)], energy=np.array([0.0, 1.0, np.nan, 2.0, 1.0]))
    assert fes.cv_labels == ["CV1"]  # generated when nothing better is known
    assert fes.label == r"$F$"
    assert fes.finite_range() == (0.0, 2.0)


# ---------------------------------------------------------------------------
# Barrier and basin analysis
# ---------------------------------------------------------------------------
def double_well(n_bins=601):
    """A profile whose numbers are known exactly, in eV.

    Two parabolas meeting at their tops: basin A bottoms out at 0.000 eV at
    x = 1, basin B at 0.200 eV at x = 5, and the join at x = 3 sits at 0.500
    eV from both sides. Basin B is the wider of the two, so it is the one
    Boltzmann weighting favours.
    """
    x = np.linspace(0.0, 6.0, n_bins)
    energy = np.where(
        x < 3.0, 0.5 * ((x - 1.0) / 2.0) ** 2, 0.2 + 0.3 * ((x - 5.0) / 2.0) ** 2
    )
    return np.column_stack([x, energy])


@pytest.fixture
def well():
    return double_well()


class TestSummariseFes:
    def test_finds_both_minima(self, well):
        summary = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert summary.minimum_a == pytest.approx(1.0, abs=0.01)
        assert summary.minimum_b == pytest.approx(5.0, abs=0.01)

    def test_reports_the_depth_of_each_basin(self, well):
        summary = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert summary.depth_a == pytest.approx(0.0, abs=1e-4)
        assert summary.depth_b == pytest.approx(0.2, abs=1e-4)

    def test_finds_the_barrier_between_them(self, well):
        summary = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert summary.barrier_position == pytest.approx(3.0, abs=0.01)
        assert summary.forward_barrier == pytest.approx(0.5, abs=1e-3)
        assert summary.reverse_barrier == pytest.approx(0.3, abs=1e-3)

    def test_reports_the_basin_difference(self, well):
        summary = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert summary.delta_f == pytest.approx(0.2, abs=1e-4)

    def test_swapping_the_basins_flips_the_sign(self, well):
        forward = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")
        backward = summarise_fes(well, (4.0, 6.0), (0.0, 2.0), source_unit="eV")

        assert backward.delta_f == pytest.approx(-forward.delta_f)
        assert backward.forward_barrier == pytest.approx(forward.reverse_barrier)
        assert backward.reverse_barrier == pytest.approx(forward.forward_barrier)

    def test_a_basin_may_be_given_either_way_round(self, well):
        one = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")
        other = summarise_fes(well, (2.0, 0.0), (6.0, 4.0), source_unit="eV")

        assert other.delta_f == pytest.approx(one.delta_f)

    def test_boltzmann_weighting_favours_the_wider_basin(self, well):
        """Basin B is the flatter one, so entropy pulls its free energy down."""
        depths = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")
        weighted = summarise_fes(
            well, (0.0, 2.0), (4.0, 6.0), source_unit="eV", temperature=300
        )

        assert weighted.delta_f < depths.delta_f

    def test_temperature_leaves_the_barriers_alone(self, well):
        """A barrier is measured out of the bottom of a well either way."""
        plain = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")
        weighted = summarise_fes(
            well, (0.0, 2.0), (4.0, 6.0), source_unit="eV", temperature=300
        )

        assert weighted.forward_barrier == pytest.approx(plain.forward_barrier)
        assert weighted.reverse_barrier == pytest.approx(plain.reverse_barrier)

    def test_converts_units_on_the_way_in(self, well):
        in_ev = summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")
        in_kj = summarise_fes(
            well, (0.0, 2.0), (4.0, 6.0), source_unit="eV", energy_unit="kJ/mol"
        )

        assert in_kj.forward_barrier == pytest.approx(
            in_ev.forward_barrier * EV_IN_KJ_PER_MOL
        )
        assert in_kj.energy_unit == "kJ/mol"

    def test_reads_a_file(self, tmp_path, well):
        path = tmp_path / "profile.dat"
        np.savetxt(path, well, header="! FIELDS cv file.free", comments="#")

        summary = summarise_fes(str(path), (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert summary.forward_barrier == pytest.approx(0.5, abs=1e-3)

    def test_ignores_unsampled_bins(self, well):
        """NaNs are holes in the surface, not zero-energy points."""
        holed = well.copy()
        holed[well[:, 0] > 5.5, 1] = np.nan

        summary = summarise_fes(holed, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert summary.depth_b == pytest.approx(0.2, abs=1e-4)

    def test_prints_a_readable_summary(self, well):
        text = str(summarise_fes(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV"))

        assert "Barrier A->B:  0.500 eV" in text
        assert "Barrier B->A:  0.300 eV" in text
        assert "Delta F (B-A): 0.200 eV" in text

    def test_rejects_overlapping_basins(self, well):
        with pytest.raises(ValueError, match="overlap"):
            summarise_fes(well, (0.0, 2.0), (1.0, 3.0), source_unit="eV")

    def test_rejects_basins_with_nothing_between_them(self, well):
        with pytest.raises(ValueError, match="between the basins"):
            summarise_fes(well, (0.0, 2.9), (2.91, 6.0), source_unit="eV")

    def test_rejects_a_basin_off_the_grid(self, well):
        with pytest.raises(ValueError, match="no sampled grid point"):
            summarise_fes(well, (20.0, 30.0), (4.0, 6.0), source_unit="eV")

    def test_rejects_a_basin_of_only_unsampled_bins(self, well):
        holed = well.copy()
        holed[well[:, 0] <= 2.0, 1] = np.nan  # the window is inclusive

        with pytest.raises(ValueError, match="no sampled grid point"):
            summarise_fes(holed, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

    def test_rejects_a_two_dimensional_surface(self, fes_2d_file):
        with pytest.raises(ValueError, match="1-D profile"):
            summarise_fes(str(fes_2d_file), (0.0, 1.0), (2.0, 3.0))

    def test_temperature_needs_a_known_energy_unit(self, well):
        with pytest.raises(ValueError, match="without knowing"):
            summarise_fes(
                well, (0.0, 2.0), (4.0, 6.0), source_unit=None, temperature=300
            )


class TestFesConvergence:
    @pytest.fixture
    def series(self, well):
        """A barrier growing towards its final height, as hills fill a well."""
        return [
            np.column_stack([well[:, 0], well[:, 1] * scale])
            for scale in (0.3, 0.6, 0.85, 1.0)
        ]

    def test_summarises_every_surface(self, series):
        summaries = fes_convergence(series, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert len(summaries) == len(series)
        assert all(isinstance(s, FESSummary) for s in summaries)

    def test_keeps_the_order_it_was_given(self, series):
        summaries = fes_convergence(series, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        barriers = [s.forward_barrier for s in summaries]
        assert barriers == sorted(barriers)
        assert barriers[-1] == pytest.approx(0.5, abs=1e-3)

    def test_uses_the_same_basins_throughout(self, series):
        """Fixed windows are what make the numbers comparable across a run."""
        summaries = fes_convergence(series, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert len({s.minimum_a for s in summaries}) == 1

    def test_accepts_a_single_surface(self, well):
        assert len(fes_convergence(well, (0.0, 2.0), (4.0, 6.0), source_unit="eV")) == 1


class TestPlotFesConvergence:
    @pytest.fixture
    def series(self, well):
        return [
            np.column_stack([well[:, 0], well[:, 1] * scale])
            for scale in (0.3, 0.6, 0.85, 1.0)
        ]

    def test_draws_the_barrier_and_the_difference(self, series):
        _, ax = plot_fes_convergence(series, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert len(ax.lines) == 2
        assert len(ax.get_legend().get_texts()) == 2

    def test_numbers_the_surfaces_when_given_no_times(self, series):
        _, ax = plot_fes_convergence(series, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert list(ax.lines[0].get_xdata()) == [1, 2, 3, 4]

    def test_uses_the_times_it_is_given(self, series):
        _, ax = plot_fes_convergence(
            series, (0.0, 2.0), (4.0, 6.0), source_unit="eV", times=[25, 50, 75, 100]
        )

        assert list(ax.lines[0].get_xdata()) == [25, 50, 75, 100]

    def test_the_barrier_curve_matches_the_summaries(self, series):
        summaries = fes_convergence(series, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        _, ax = plot_fes_convergence(series, (0.0, 2.0), (4.0, 6.0), source_unit="eV")

        assert list(ax.lines[0].get_ydata()) == pytest.approx(
            [s.forward_barrier for s in summaries]
        )

    def test_checks_the_number_of_times(self, series):
        with pytest.raises(ValueError, match="2 times for 4 surfaces"):
            plot_fes_convergence(
                series, (0.0, 2.0), (4.0, 6.0), source_unit="eV", times=[1, 2]
            )

    def test_draws_on_a_supplied_axes(self, series):
        fig, ax = plt.subplots()

        returned_fig, returned_ax = plot_fes_convergence(
            series, (0.0, 2.0), (4.0, 6.0), source_unit="eV", fig=fig, ax=ax
        )

        assert returned_fig is fig
        assert returned_ax is ax

    def test_saves_when_given_a_filename(self, tmp_path, series):
        plot_fes_convergence(
            series,
            (0.0, 2.0),
            (4.0, 6.0),
            source_unit="eV",
            filename=str(tmp_path / "conv.png"),
        )

        assert (tmp_path / "conv.png").exists()


class TestFesSeriesFiles:
    """Finding a numbered convergence series, in the order it was written."""

    def _write_series(self, directory, names):
        for name in names:
            path = directory / name
            np.savetxt(path, np.column_stack([np.linspace(-1, 1, 5), np.zeros(5)]))

    def test_files_come_back_in_index_order_not_alphabetical(self, tmp_path):
        # fes_10 sorts before fes_2 by name, which scrambles a convergence
        # series -- where the order is the entire point.
        self._write_series(tmp_path, ["fes_1.dat", "fes_2.dat", "fes_10.dat"])

        found = fes_series_files(tmp_path)

        assert [path.name for path in found] == ["fes_1.dat", "fes_2.dat", "fes_10.dat"]

    def test_the_underscore_is_optional(self, tmp_path):
        self._write_series(tmp_path, ["fes1.dat", "fes2.dat"])

        assert len(fes_series_files(tmp_path)) == 2

    def test_matching_is_case_insensitive(self, tmp_path):
        self._write_series(tmp_path, ["FES1.dat"])

        assert len(fes_series_files(tmp_path)) == 1

    def test_unnumbered_and_unrelated_files_are_left_out(self, tmp_path):
        self._write_series(tmp_path, ["fes_1.dat", "fes.dat", "COLVAR", "HILLS"])

        assert [path.name for path in fes_series_files(tmp_path)] == ["fes_1.dat"]

    def test_an_empty_directory_gives_an_empty_list(self, tmp_path):
        assert fes_series_files(tmp_path) == []

    def test_a_custom_pattern_is_honoured(self, tmp_path):
        self._write_series(tmp_path, ["surface_3.dat", "surface_1.dat"])

        found = fes_series_files(tmp_path, pattern=r"^surface_(\d+)\.dat$")

        assert [path.name for path in found] == ["surface_1.dat", "surface_3.dat"]


class TestLoadFesSeries:
    def test_it_loads_every_surface_in_order(self, tmp_path):
        for i in (1, 2, 10):
            np.savetxt(
                tmp_path / f"fes_{i}.dat",
                np.column_stack([np.linspace(-1, 1, 5), np.full(5, float(i))]),
            )

        series = load_fes_series(tmp_path, verbose=False)

        assert len(series) == 3
        assert all(fes.ndim == 1 for fes in series)

    def test_energies_are_converted_out_of_the_source_unit(self, tmp_path):
        # 1 eV is 96.485 kJ/mol; written in kJ/mol and asked for in eV, the
        # spread of the surface should come back divided by that.
        np.savetxt(
            tmp_path / "fes_1.dat",
            np.column_stack(
                [np.linspace(-1, 1, 5), np.array([96.48533212331, 0, 0, 0, 0])]
            ),
        )

        fes = load_fes_series(
            tmp_path, energy_unit="eV", source_unit="kJ/mol", verbose=False
        )[0]

        assert np.nanmax(fes.energy) == pytest.approx(1.0)

    def test_an_empty_directory_gives_an_empty_list(self, tmp_path):
        assert load_fes_series(tmp_path, verbose=False) == []
