"""Tools for transition-state, NEB and metadynamics calculations.

The package is organised into four modules, all of which are re-exported here:

``tools_reaction``
    Build, optimise and post-process nudged elastic band (NEB) paths, either
    one image at a time or with the images spread over socket calculators,
    refine the top of a band into a true saddle point and follow the intrinsic
    reaction coordinate away from it, and take quick geodesic guesses at a
    path or a transition state.
``tools_geometry``
    Work out which atoms make up each half of a stacked dimer and swap those
    halves over, to build a flipped end state for a band.
``tools_plumed``
    Prepare PLUMED input and turn metadynamics hills into a free-energy
    surface.
``tools_plotting``
    Consistent matplotlib styling plus ready-made plots for structures, NEB
    and IRC profiles, MD trajectories and PLUMED free-energy surfaces.

The saddle-point searches (``optimise_ts`` and ``optimise_irc``) are built on
sella, which is an optional dependency: install it with
``pip install 'reactiontools[ts]'``. Everything else works without it.
"""

from .tools_geometry import (bonded_cluster_indices_no_anchor_hub,
                             get_dimer_bonded_cluster_indices,
                             flip_and_face_bases,
                             optimize_with_fixed_anchors,
                             get_best_flip_and_face_bases)
from .tools_plotting import (n_plot,
                             ax_plot,
                             show_atoms,
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
from .tools_reaction import (get_neb_path,
                             get_fmax,
                             stitch_path,
                             resample_path,
                             optimise_geom,
                             optimise_reactant_product,
                             prepare_neb,
                             socket_calculators,
                             prepare_parallel_neb,
                             optimise_neb,
                             get_ts_image,
                             optimise_ts,
                             optimise_irc,
                             get_vibrations,
                             quick_guess_path,
                             quick_guess_ts)

__version__ = "0.1.0"

__all__ = [
    # tools_reaction
    "get_neb_path",
    "get_fmax",
    "stitch_path",
    "resample_path",
    "optimise_geom",
    "optimise_reactant_product",
    "prepare_neb",
    "socket_calculators",
    "prepare_parallel_neb",
    "optimise_neb",
    "get_ts_image",
    "optimise_ts",
    "optimise_irc",
    "get_vibrations",
    "quick_guess_path",
    "quick_guess_ts",
    # tools_geometry
    "bonded_cluster_indices_no_anchor_hub",
    "get_dimer_bonded_cluster_indices",
    "flip_and_face_bases",
    "optimize_with_fixed_anchors",
    "get_best_flip_and_face_bases",
    # tools_plumed
    "plumed_selection",
    "find_molecules",
    "run_sum_hills",
    # tools_plotting
    "n_plot",
    "ax_plot",
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
