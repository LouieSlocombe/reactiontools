"""Package-level tests: the public surface and its optional dependencies."""

import importlib.util

import pytest

import reactiontools

HAS_GEODESIC = importlib.util.find_spec("geodesic_interpolate") is not None


def test_version_is_a_string():
    assert isinstance(reactiontools.__version__, str)
    assert reactiontools.__version__.count(".") == 2


def test_all_names_are_importable():
    """Everything advertised in __all__ must actually be re-exported."""
    missing = [name for name in reactiontools.__all__
               if not hasattr(reactiontools, name)]
    assert not missing


def test_public_api_is_complete():
    """The documented API must all be reachable from the top-level package."""
    expected = {
        # tools_reaction
        "get_neb_path", "stitch_path", "resample_path", "optimise_geom",
        "optimise_reactant_product", "prepare_neb", "optimise_neb",
        "get_ts_image", "quick_guess_path", "quick_guess_ts",
        # tools_plumed
        "plumed_selection", "find_molecules", "run_sum_hills",
        # tools_plotting
        "n_plot", "ax_plot", "plot_images", "plot_neb", "plot_temperature",
        "plot_total_energy", "plot_plumed", "plot_plumed_multi",
    }
    assert expected <= set(reactiontools.__all__)


def test_package_imports_without_geodesic_interpolate():
    """The git-only dependency must not be needed to import the package.

    tools_reaction imports it lazily, so the modules that do not touch
    geodesic interpolation stay usable when it is absent.
    """
    from reactiontools import tools_reaction

    assert not hasattr(tools_reaction, "gi")


@pytest.mark.skipif(HAS_GEODESIC, reason="geodesic_interpolate is installed")
def test_geodesic_functions_raise_a_helpful_error(water):
    """Without the dependency, the error should say how to install it."""
    with pytest.raises(ImportError, match="geodesic_interpolate"):
        reactiontools.quick_guess_ts(water, water, n_images=5)


@pytest.mark.skipif(not HAS_GEODESIC, reason="requires geodesic_interpolate")
def test_quick_guess_path_returns_the_requested_images(water):
    product = water.copy()
    product.positions[1] += [0.4, 0.0, 0.0]

    path = reactiontools.quick_guess_path(water, product, n_images=7)

    assert len(path) == 7


@pytest.mark.skipif(not HAS_GEODESIC, reason="requires geodesic_interpolate")
def test_quick_guess_ts_returns_a_midpoint_structure(water):
    product = water.copy()
    product.positions[1] += [0.4, 0.0, 0.0]

    ts = reactiontools.quick_guess_ts(water, product, n_images=7)

    assert len(ts) == len(water)
