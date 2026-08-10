"""Tests for reactiontools.tools_plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from reactiontools import (ax_plot,
                           n_plot,
                           plot_images,
                           plot_irc,
                           plot_neb,
                           plot_plumed,
                           plot_plumed_multi,
                           plot_temperature,
                           plot_total_energy,
                           prepare_neb,
                           show_atoms,
                           summarise_neb)
from reactiontools.tools_plotting import (_expand_fes_files,
                                          _fes_labels,
                                          _get_energy)


@pytest.fixture
def band(calc, water):
    """A short, energy-evaluated NEB band."""
    product = water.copy()
    product.positions[1] += [0.4, 0.0, 0.0]
    neb = prepare_neb(water, product, calc, n_images=5, geo_int=False)
    return list(neb.images)


class TestStyleHelpers:
    def test_n_plot_labels_the_current_axes(self):
        fig, ax = plt.subplots()
        plt.sca(ax)

        n_plot("Path (Å)", "Energy (meV)")

        assert ax.get_xlabel() == "Path (Å)"
        assert ax.get_ylabel() == "Energy (meV)"

    def test_ax_plot_labels_the_given_axes(self):
        fig, (left, right) = plt.subplots(1, 2)

        ax_plot(fig, right, "x", "y")

        assert right.get_xlabel() == "x"
        assert left.get_xlabel() == ""

    def test_ax_plot_applies_the_font_sizes(self):
        fig, ax = plt.subplots()

        ax_plot(fig, ax, "x", "y", xs=20, ys=18)

        assert ax.xaxis.label.get_fontsize() == 20
        assert ax.yaxis.label.get_fontsize() == 18


class TestPlotImages:
    def test_one_panel_per_image(self, chain):
        fig, axes = plot_images(chain, show=False)

        assert len(axes) >= len(chain)

    def test_accepts_a_single_atoms_object(self, water):
        fig, axes = plot_images(water, show=False)

        assert len(axes) == 1

    def test_titles_default_to_the_image_index(self, chain):
        fig, axes = plot_images(chain, show=False)

        assert [ax.get_title() for ax in axes[:len(chain)]] == ["0", "1", "2", "3", "4"]

    def test_custom_titles_are_used(self, chain):
        titles = ["R", "a", "TS", "b", "P"]

        fig, axes = plot_images(chain, titles=titles, show=False)

        assert [ax.get_title() for ax in axes[:len(chain)]] == titles

    def test_rejects_a_title_count_mismatch(self, chain):
        with pytest.raises(ValueError, match="titles"):
            plot_images(chain, titles=["only", "two"], show=False)

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValueError, match="No images"):
            plot_images([], show=False)

    def test_grid_wraps_at_n_cols(self, chain):
        fig, axes = plot_images(chain, n_cols=2, show=False)

        # 5 images at 2 per row needs 3 rows, so 6 panels
        assert len(axes) == 6

    def test_all_panels_have_their_frames_hidden(self, chain):
        fig, axes = plot_images(chain, n_cols=2, show=False)

        assert not any(ax.axison for ax in axes)

    @pytest.mark.parametrize("view", ["top", "side", "front", "tilted"])
    def test_named_views_are_accepted(self, water, view):
        fig, axes = plot_images(water, view=view, show=False)

        assert len(axes) == 1

    def test_raw_rotation_strings_are_accepted(self, water):
        fig, axes = plot_images(water, rotation="45x,10y,0z", show=False)

        assert len(axes) == 1

    def test_save_writes_png_and_pdf(self, water):
        plot_images(water, save=True, show=False, filename="frames")

        assert Path("frames.png").exists()
        assert Path("frames.pdf").exists()

    def test_does_not_save_by_default(self, water):
        plot_images(water, show=False, filename="frames")

        assert not Path("frames.png").exists()


class TestGetEnergy:
    def test_uses_the_cached_energy(self, band):
        """Cached results mean the calculator is never needed."""
        assert _get_energy(band[0], calc=None) == pytest.approx(
            band[0].get_potential_energy())

    def test_falls_back_to_the_calculator(self, calc, water):
        water.calc = None

        assert _get_energy(water, calc) == pytest.approx(
            water.get_potential_energy())


class TestPlotNeb:
    def test_returns_a_figure_and_axes(self, band, calc):
        fig, ax = plot_neb(band, calc, save=False, show=False)

        assert fig is ax.get_figure()

    def test_energies_are_referenced_to_the_lowest_image(self, band, calc):
        fig, ax = plot_neb(band, calc, smooth=False, save=False, show=False)

        assert ax.lines[0].get_ydata().min() == pytest.approx(0.0)

    def test_energies_are_converted_to_mev(self, band, calc):
        fig, ax = plot_neb(band, calc, smooth=False, save=False, show=False)

        energies = np.array([image.get_potential_energy() for image in band])
        expected = (energies - energies.min()) * 1000.0
        assert ax.lines[0].get_ydata() == pytest.approx(expected)

    def test_x_axis_is_the_path_coordinate(self, band, calc):
        fig, ax = plot_neb(band, calc, smooth=False, save=False, show=False)

        x = ax.lines[0].get_xdata()
        assert x[0] == pytest.approx(0.0)
        assert np.all(np.diff(x) >= 0)

    def test_smoothing_adds_interpolated_points(self, band, calc):
        fig, ax = plot_neb(band, calc, smooth=True, save=False, show=False)

        assert len(ax.lines[0].get_xdata()) > len(band)

    def test_does_not_annotate_by_default(self, band, calc):
        """Existing figures must not gain text they did not have."""
        fig, ax = plot_neb(band, calc, save=False, show=False)

        assert len(ax.texts) == 0

    def test_annotates_the_barrier_when_asked(self, band, calc):
        fig, ax = plot_neb(band, calc, save=False, show=False, annotate=True)

        assert len(ax.texts) == 1
        assert "meV" in ax.texts[0].get_text()

    def test_the_annotated_barrier_matches_summarise_neb(self, band, calc):
        """The figure and the numbers must not disagree."""
        fig, ax = plot_neb(band, calc, save=False, show=False, annotate=True)

        expected = summarise_neb(band, calc).barrier * 1000.0
        drawn = float(ax.texts[0].get_text().split("=")[1].replace("meV", ""))
        assert drawn == pytest.approx(expected, abs=0.5)

    def test_draws_on_a_supplied_axes(self, band, calc):
        fig, ax = plt.subplots()

        returned_fig, returned_ax = plot_neb(band, calc, fig=fig, ax=ax,
                                             save=False, show=False)

        assert returned_ax is ax
        assert returned_fig is fig

    def test_save_writes_the_supplied_figure(self, band, calc):
        """The saved file must be the figure passed in, not the active one."""
        fig, ax = plt.subplots()
        plt.figure()  # make a different figure current

        plot_neb(band, calc, fig=fig, ax=ax, save=True, show=False,
                 filename="profile")

        assert Path("profile.png").exists()
        assert Path("profile.pdf").exists()

    def test_label_is_applied(self, band, calc):
        fig, ax = plot_neb(band, calc, smooth=False, save=False, show=False,
                           label="forward")

        assert ax.lines[0].get_label() == "forward"

    def test_axis_labels_are_set(self, band, calc):
        fig, ax = plot_neb(band, calc, save=False, show=False)

        assert "Path" in ax.get_xlabel()
        assert "meV" in ax.get_ylabel()


class TestPlotTemperature:
    def test_returns_a_figure_and_axes(self, md_trajectory):
        fig, ax = plot_temperature(md_trajectory)

        assert fig is ax.get_figure()

    def test_accepts_a_supplied_axes(self, md_trajectory):
        """Regression: fig was only bound when ax was None, raising NameError."""
        fig, ax = plt.subplots()

        returned_fig, returned_ax = plot_temperature(md_trajectory, ax=ax)

        assert returned_ax is ax
        assert returned_fig is fig

    def test_plots_one_line_per_trajectory(self, md_trajectory):
        fig, ax = plot_temperature([md_trajectory, md_trajectory])

        assert len(ax.lines) == 2

    def test_labels_default_to_the_filename(self, md_trajectory):
        fig, ax = plot_temperature(md_trajectory)

        assert ax.lines[0].get_label() == "md.traj"

    def test_custom_labels_are_used(self, md_trajectory):
        fig, ax = plot_temperature(md_trajectory, labels=["300 K run"])

        assert ax.lines[0].get_label() == "300 K run"

    def test_x_axis_is_frames_without_a_timestep(self, md_trajectory):
        fig, ax = plot_temperature(md_trajectory)

        assert ax.get_xlabel() == "Frame"
        assert ax.lines[0].get_xdata() == pytest.approx([0, 1, 2, 3, 4])

    def test_timestep_switches_the_x_axis_to_time(self, md_trajectory):
        fig, ax = plot_temperature(md_trajectory, timestep=0.5)

        assert ax.get_xlabel() == "Time (fs)"
        assert ax.lines[0].get_xdata() == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

    def test_y_axis_is_temperature(self, md_trajectory):
        fig, ax = plot_temperature(md_trajectory)

        assert ax.get_ylabel() == "Temperature (K)"


class TestPlotTotalEnergy:
    def test_returns_a_figure_and_axes(self, md_trajectory):
        fig, ax = plot_total_energy(md_trajectory)

        assert fig is ax.get_figure()

    def test_accepts_a_supplied_axes(self, md_trajectory):
        """Regression: fig was only bound when ax was None, raising NameError."""
        fig, ax = plt.subplots()

        returned_fig, returned_ax = plot_total_energy(md_trajectory, ax=ax)

        assert returned_ax is ax
        assert returned_fig is fig

    def test_plots_the_trajectory_energies(self, md_trajectory):
        from ase.io import read

        fig, ax = plot_total_energy(md_trajectory)

        expected = [a.get_total_energy() for a in read(md_trajectory, index=":")]
        assert ax.lines[0].get_ydata() == pytest.approx(expected)

    def test_y_axis_is_total_energy(self, md_trajectory):
        fig, ax = plot_total_energy(md_trajectory)

        assert ax.get_ylabel() == "Total energy (eV)"

    def test_timestep_switches_the_x_axis_to_time(self, md_trajectory):
        fig, ax = plot_total_energy(md_trajectory, timestep=2.0)

        assert ax.get_xlabel() == "Time (fs)"
        assert ax.lines[0].get_xdata() == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])


class TestFesHelpers:
    def test_expand_accepts_a_single_file(self, fes_file):
        assert _expand_fes_files(fes_file) == [fes_file]

    def test_expand_finds_fes_files_under_a_directory(self, tmp_path):
        for name in ("run_a", "run_b"):
            run = tmp_path / "runs" / name
            run.mkdir(parents=True)
            np.savetxt(run / "fes.dat", np.zeros((3, 2)))

        found = _expand_fes_files(tmp_path / "runs")

        assert [f.parent.name for f in found] == ["run_a", "run_b"]

    def test_expand_rejects_a_directory_with_no_surfaces(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="no fes.dat files found"):
            _expand_fes_files(empty)

    def test_labels_use_the_file_stem(self, tmp_path):
        files = [tmp_path / "cold.dat", tmp_path / "hot.dat"]

        assert _fes_labels(files) == ["cold", "hot"]

    def test_labels_fall_back_to_the_run_directory(self, tmp_path):
        """The <run>/fes.dat layout gives every file the same stem."""
        files = [tmp_path / "run_a" / "fes.dat", tmp_path / "run_b" / "fes.dat"]

        assert _fes_labels(files) == ["run_a", "run_b"]


class TestPlotPlumed:
    def test_returns_a_figure_and_axes(self, fes_file):
        fig, ax = plot_plumed(fes_file, save=False, show=False)

        assert fig is ax.get_figure()

    def test_plots_the_surface_in_mev(self, fes_file):
        """The fixture writes a cv**2 parabola in eV; meV is 1000x that.

        This is the seam between the two unit conventions: sum_hills driven
        from ASE writes eV, and these wrappers plot meV, so the conversion
        going through tools_fes has to come out at exactly 1000.
        """
        fig, ax = plot_plumed(fes_file, save=False, show=False)

        cv = np.loadtxt(fes_file, usecols=0)
        assert ax.lines[0].get_ydata() == pytest.approx(cv ** 2 * 1000.0)

    def test_does_not_shift_the_surface(self, fes_file):
        """sum_hills is normally run with --mintozero; don't shift again."""
        fig, ax = plot_plumed(fes_file, save=False, show=False)

        assert ax.lines[0].get_ydata().min() == pytest.approx(0.0)

    def test_x_range_is_applied(self, fes_file):
        fig, ax = plot_plumed(fes_file, save=False, show=False, x_range=(-0.5, 0.5))

        assert ax.get_xlim() == pytest.approx((-0.5, 0.5))

    def test_x_label_is_applied(self, fes_file):
        fig, ax = plot_plumed(fes_file, save=False, show=False, x_label="d (Å)")

        assert ax.get_xlabel() == "d (Å)"

    def test_save_writes_png_and_pdf(self, fes_file):
        plot_plumed(fes_file, save=True, show=False, filename="surface")

        assert Path("surface.png").exists()
        assert Path("surface.pdf").exists()


class TestPlotPlumedMulti:
    @pytest.fixture
    def runs(self, tmp_path):
        """Two runs laid out as <run>/fes.dat, offset from each other."""
        cv = np.linspace(-1.0, 1.0, 21)
        for name, offset in (("run_a", 0.0), ("run_b", 2.0)):
            run = tmp_path / "runs" / name
            run.mkdir(parents=True)
            np.savetxt(run / "fes.dat", np.column_stack([cv, cv ** 2 + offset]))
        return tmp_path / "runs"

    def test_one_line_per_run(self, runs):
        fig, ax = plot_plumed_multi(runs, save=False, show=False)

        assert len(ax.lines) == 2

    def test_labels_come_from_the_run_directories(self, runs):
        fig, ax = plot_plumed_multi(runs, save=False, show=False)

        assert [line.get_label() for line in ax.lines] == ["run_a", "run_b"]

    def test_custom_labels_are_used(self, runs):
        fig, ax = plot_plumed_multi(runs, labels=["cold", "hot"],
                                    save=False, show=False)

        assert [line.get_label() for line in ax.lines] == ["cold", "hot"]

    def test_mintozero_shifts_each_curve_independently(self, runs):
        fig, ax = plot_plumed_multi(runs, mintozero=True, save=False, show=False)

        for line in ax.lines:
            assert line.get_ydata().min() == pytest.approx(0.0)

    def test_offsets_are_kept_without_mintozero(self, runs):
        fig, ax = plot_plumed_multi(runs, mintozero=False, save=False, show=False)

        minima = [line.get_ydata().min() for line in ax.lines]
        assert minima[1] - minima[0] == pytest.approx(2000.0)

    def test_runs_get_different_colours(self, runs):
        fig, ax = plot_plumed_multi(runs, save=False, show=False)

        assert ax.lines[0].get_color() != ax.lines[1].get_color()

    def test_accepts_an_explicit_file_list(self, runs):
        files = sorted(runs.glob("*/fes.dat"))

        fig, ax = plot_plumed_multi(files, save=False, show=False)

        assert len(ax.lines) == 2

    def test_save_writes_png_and_pdf(self, runs):
        plot_plumed_multi(runs, save=True, show=False, filename="compare")

        assert Path("compare.png").exists()
        assert Path("compare.pdf").exists()


class TestShowAtoms:
    def test_accepts_a_single_structure(self, water):
        fig, ax = show_atoms(water, save=False, show=False)

        assert fig is not None and ax is not None

    def test_overlays_a_whole_band_on_one_axes(self, band):
        """The point of it: one frame, every image drawn into it."""
        _fig, ax = show_atoms(band, save=False, show=False)

        single = plt.subplots()[1]
        show_atoms(band[0], ax=single, save=False, show=False)
        assert len(ax.patches) > len(single.patches)

    def test_draws_on_a_given_axes(self, water):
        _fig, ax = plt.subplots()

        _fig2, used = show_atoms(water, ax=ax, save=False, show=False)

        assert used is ax

    def test_saves_both_formats(self, water, tmp_path):
        show_atoms(water, save=True, show=False, filename="struct")

        assert (tmp_path / "struct.png").exists()
        assert (tmp_path / "struct.pdf").exists()

    def test_accepts_a_named_view(self, water):
        fig, _ax = show_atoms(water, view="side", save=False, show=False)

        assert fig is not None


class TestPlotIrc:
    def test_returns_a_figure_and_axes(self, band):
        fig, ax = plot_irc(band, save=False, show=False)

        assert fig is not None and ax is not None

    def test_uses_the_energies_the_images_carry(self, band):
        """No calculator needed for a path read back from a trajectory."""
        fig, _ax = plot_irc(band, calc=None, save=False, show=False)

        assert fig is not None

    def test_plots_in_mev_above_the_minimum(self, band):
        _fig, ax = plot_irc(band, save=False, show=False, smooth=False)

        drawn = ax.lines[0].get_ydata()
        energies = np.array([_get_energy(i, None) for i in band])
        assert drawn == pytest.approx((energies - energies.min()) * 1000.0)

    def test_composes_onto_an_existing_axes(self, band):
        fig, ax = plt.subplots()

        _fig, used = plot_irc(band, fig=fig, ax=ax, save=False, show=False)

        assert used is ax

    def test_saves_both_formats(self, band, tmp_path):
        plot_irc(band, save=True, show=False, filename="path")

        assert (tmp_path / "path.png").exists()
        assert (tmp_path / "path.pdf").exists()

    def test_labels_the_axes_in_mev(self, band):
        _fig, ax = plot_irc(band, save=False, show=False)

        assert "meV" in ax.get_ylabel()
