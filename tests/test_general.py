"""Package-level tests: the public surface."""

import reactiontools


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
        "ConvergenceError", "ConvergenceWarning", "NebSummary",
        "summarise_neb", "restart_neb", "restart_parallel_neb",
        "get_neb_path", "get_fmax", "stitch_path", "resample_path",
        "optimise_geom", "optimise_reactant_product", "prepare_neb",
        "socket_calculators", "prepare_parallel_neb", "optimise_neb",
        "get_ts_image", "optimise_ts", "optimise_irc", "get_vibrations",
        "quick_guess_path", "quick_guess_ts",
        # tools_geometry
        "bonded_cluster_indices_no_anchor_hub",
        "get_dimer_bonded_cluster_indices", "flip_and_face_bases",
        "optimize_with_fixed_anchors", "get_best_flip_and_face_bases",
        # tools_plumed
        "plumed_selection", "find_molecules", "run_sum_hills",
        # tools_plotting
        "n_plot", "ax_plot", "show_atoms", "plot_images", "plot_neb",
        "plot_irc", "plot_temperature", "plot_total_energy", "plot_plumed",
        "plot_plumed_multi",
    }
    assert expected <= set(reactiontools.__all__)


def test_quick_guess_path_returns_the_requested_images(water):
    product = water.copy()
    product.positions[1] += [0.4, 0.0, 0.0]

    path = reactiontools.quick_guess_path(water, product, n_images=7)

    assert len(path) == 7


def test_quick_guess_ts_returns_a_midpoint_structure(water):
    product = water.copy()
    product.positions[1] += [0.4, 0.0, 0.0]

    ts = reactiontools.quick_guess_ts(water, product, n_images=7)

    assert len(ts) == len(water)
