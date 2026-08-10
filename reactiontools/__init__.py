"""Tools for transition-state, NEB and metadynamics calculations.

The package is organised into six modules, all of which are re-exported here:

``tools_reaction``
    Build, optimise and post-process nudged elastic band (NEB) paths, either
    one image at a time or with the images spread over socket calculators,
    continue a band that has already been relaxed once, refine the top of a
    band into a true saddle point and follow the intrinsic reaction coordinate
    away from it, and take quick geodesic guesses at a path or a transition
    state.
``tools_orca``
    Build ASE ORCA calculators from a few presets, optimise a geometry with
    ORCA's own driver, and run a GOAT conformer search.
``tools_geometry``
    Work out which atoms make up each half of a stacked dimer and swap those
    halves over, or move a hydrogen across a hydrogen bond, to build a product
    end state for a band.
``tools_plumed``
    Prepare PLUMED input and turn metadynamics hills into a free-energy
    surface.
``tools_fes``
    Read PLUMED output -- ``COLVAR``, ``fes.dat``, ``HILLS`` -- and plot free
    energy in one or two dimensions, in whichever energy unit you want.
``tools_plotting``
    Ready-made plots for structures, NEB and IRC profiles, MD trajectories and
    one-dimensional PLUMED surfaces.

Consistent matplotlib styling lives in ``tools_style`` and is applied by every
plotting function; ``n_plot`` and ``ax_plot`` are exported for use on your own
axes.

Every ``optimise_*`` function records whether it actually reached its force
criterion in ``info["converged"]`` on the structures it returns, and warns
``ConvergenceWarning`` when it did not. Pass ``raise_on_unconverged=True`` for
a ``ConvergenceError`` instead, or promote every one of them at once with
``warnings.simplefilter("error", ConvergenceWarning)``.

Two dependencies are needed only by part of the package. The saddle-point
searches (``optimise_ts`` and ``optimise_irc``) are built on sella, an optional
extra: install it with ``pip install 'reactiontools[ts]'``. The ``tools_orca``
functions shell out to ORCA, which is licensed separately and installed by
hand; see ``build_tools/README.md``. Everything else works without either.
"""

from .tools_fes import (DEFAULT_ENERGY_UNIT,
                        ENERGY_UNITS,
                        FES,
                        PlumedData,
                        as_fes,
                        convert_energy,
                        plot_fes,
                        plot_fes_1d,
                        plot_fes_2d,
                        plot_fes_2d_overlay,
                        plot_fes_slices,
                        plot_plumed_colvar,
                        plot_plumed_fes,
                        read_plumed_file,
                        unit_label)
from .tools_geometry import (bonded_cluster_indices_no_anchor_hub,
                             get_dimer_bonded_cluster_indices,
                             flip_and_face_bases,
                             optimize_with_fixed_anchors,
                             get_best_flip_and_face_bases,
                             swap_bonding_configuration)
from .tools_orca import (orca_calc_preset,
                         orca_optimise_atoms,
                         orca_calculate_goat)
from .tools_plotting import (show_atoms,
                             plot_images,
                             plot_neb,
                             plot_irc,
                             plot_temperature,
                             plot_total_energy,
                             plot_plumed,
                             plot_plumed_multi)
from .tools_plumed import (plumed_selection,
                           find_molecules,
                           run_sum_hills)
from .tools_reaction import (ConvergenceError,
                             ConvergenceWarning,
                             get_neb_path,
                             get_fmax,
                             stitch_path,
                             resample_path,
                             optimise_geom,
                             optimise_reactant_product,
                             prepare_neb,
                             restart_neb,
                             socket_calculators,
                             prepare_parallel_neb,
                             restart_parallel_neb,
                             optimise_neb,
                             get_ts_image,
                             optimise_ts,
                             optimise_irc,
                             get_vibrations,
                             quick_guess_path,
                             quick_guess_ts)
from .tools_style import n_plot, ax_plot

__version__ = "0.1.0"

__all__ = [
    # tools_reaction
    "ConvergenceError",
    "ConvergenceWarning",
    "get_neb_path",
    "get_fmax",
    "stitch_path",
    "resample_path",
    "optimise_geom",
    "optimise_reactant_product",
    "prepare_neb",
    "restart_neb",
    "socket_calculators",
    "prepare_parallel_neb",
    "restart_parallel_neb",
    "optimise_neb",
    "get_ts_image",
    "optimise_ts",
    "optimise_irc",
    "get_vibrations",
    "quick_guess_path",
    "quick_guess_ts",
    # tools_orca
    "orca_calc_preset",
    "orca_optimise_atoms",
    "orca_calculate_goat",
    # tools_geometry
    "bonded_cluster_indices_no_anchor_hub",
    "get_dimer_bonded_cluster_indices",
    "flip_and_face_bases",
    "optimize_with_fixed_anchors",
    "get_best_flip_and_face_bases",
    "swap_bonding_configuration",
    # tools_plumed
    "plumed_selection",
    "find_molecules",
    "run_sum_hills",
    # tools_fes
    "DEFAULT_ENERGY_UNIT",
    "ENERGY_UNITS",
    "FES",
    "PlumedData",
    "as_fes",
    "convert_energy",
    "plot_fes",
    "plot_fes_1d",
    "plot_fes_2d",
    "plot_fes_2d_overlay",
    "plot_fes_slices",
    "plot_plumed_colvar",
    "plot_plumed_fes",
    "read_plumed_file",
    "unit_label",
    # tools_style
    "n_plot",
    "ax_plot",
    # tools_plotting
    "show_atoms",
    "plot_images",
    "plot_neb",
    "plot_irc",
    "plot_temperature",
    "plot_total_energy",
    "plot_plumed",
    "plot_plumed_multi",
    "__version__",
]
