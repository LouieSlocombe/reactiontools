"""Tools for transition-state, NEB and metadynamics calculations.

The package is organised into three modules, all of which are re-exported here:

``tools_reaction``
    Build, optimise and post-process nudged elastic band (NEB) paths, and take
    quick geodesic guesses at a path or a transition state.
``tools_plumed``
    Prepare PLUMED input and turn metadynamics hills into a free-energy
    surface.
``tools_plotting``
    Consistent matplotlib styling plus ready-made plots for structures, NEB
    profiles, MD trajectories and PLUMED free-energy surfaces.
"""

from .tools_plotting import (n_plot,
                             ax_plot,
                             plot_images,
                             plot_neb,
                             plot_temperature,
                             plot_total_energy,
                             plot_plumed,
                             plot_plumed_multi)
from .tools_plumed import (plumed_selection,
                           find_molecules,
                           run_sum_hills)
from .tools_reaction import (get_neb_path,
                             stitch_path,
                             resample_path,
                             optimise_geom,
                             optimise_reactant_product,
                             prepare_neb,
                             optimise_neb,
                             get_ts_image,
                             quick_guess_path,
                             quick_guess_ts)

__version__ = "0.0.1"

__all__ = [
    # tools_reaction
    "get_neb_path",
    "stitch_path",
    "resample_path",
    "optimise_geom",
    "optimise_reactant_product",
    "prepare_neb",
    "optimise_neb",
    "get_ts_image",
    "quick_guess_path",
    "quick_guess_ts",
    # tools_plumed
    "plumed_selection",
    "find_molecules",
    "run_sum_hills",
    # tools_plotting
    "n_plot",
    "ax_plot",
    "plot_images",
    "plot_neb",
    "plot_temperature",
    "plot_total_energy",
    "plot_plumed",
    "plot_plumed_multi",
    "__version__",
]
