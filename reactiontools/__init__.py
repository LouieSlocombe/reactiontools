"""Tools for transition-state, NEB and metadynamics calculations.

The package is organised into ten modules, all of which are re-exported here:

``tools_reaction``
    Build, optimise and post-process nudged elastic band (NEB) paths, either
    one image at a time or with the images spread over socket calculators,
    continue a band that has already been relaxed once, refine the top of a
    band into a true saddle point and follow the intrinsic reaction coordinate
    away from it, and take quick geodesic guesses at a path or a transition
    state.
``tools_orca``
    Build ASE ORCA calculators from a few presets -- named levels of theory in
    the ``orca_preset_*`` dictionaries -- optimise a geometry with ORCA's own
    driver, and run a GOAT conformer search.
``tools_geometry``
    Work out which atoms make up each half of a stacked dimer and swap those
    halves over, or move a hydrogen across a hydrogen bond, to build a product
    end state for a band.
``tools_io``
    Read and write the structure files a reaction path passes through -- XYZ,
    PDB, and the multi-model reference a ``PATHMSD`` collective variable is
    built from.
``tools_path``
    Turn a steered-MD trajectory into that reference path, by picking frames
    evenly spaced along a collective variable, and size the ``LAMBDA`` it
    should be biased with.
``tools_cv``
    Collective variables for proton transfer -- one or two transfers, or
    progress along a reference path -- and the PLUMED input that biases them,
    plus the public plumbing for writing a collective variable of your own.
``tools_plumed``
    Prepare PLUMED input, bias an ASE dynamics run with it, and turn the
    metadynamics hills that come out into a free-energy surface.
``tools_fes``
    Read PLUMED output -- ``COLVAR``, ``fes.dat``, ``HILLS`` -- and plot free
    energy in one or two dimensions, in whichever energy unit you want.
``tools_units``
    The energy units the rest of the package converts between, and the thermal
    energy kBT that reweighting needs.
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

Sella and MDTraj are installed dependencies but are imported only by the
workflows that use them. The ``tools_orca`` functions shell out to ORCA, which
is licensed separately and installed by hand; see ``build_tools/README.md``.
"""

from .tools_cv import (
    as_positions,
    plumed_angle_radians,
    plumed_bias_and_fes,
    plumed_input_1pt,
    plumed_input_2pt_1d,
    plumed_input_2pt_2d,
    plumed_input_neb_path,
    plumed_input_steered,
    plumed_input_steered_pt,
    plumed_one_based,
    plumed_temperature_pair,
    plumed_units_header,
    switching_value,
)
from .tools_fes import (
    FES,
    FESSummary,
    PlumedData,
    as_fes,
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
)
from .tools_geometry import (
    bonded_cluster_indices_no_anchor_hub,
    flip_and_face_bases,
    get_best_flip_and_face_bases,
    get_dimer_bonded_cluster_indices,
    optimize_with_fixed_anchors,
    swap_bonding_configuration,
)
from .tools_io import (
    convert_pdb_to_xyz,
    convert_xyz_to_pdb,
    convert_xyz_to_plumed_ref,
    element_from_pdb_line,
    format_pdb_atom_name,
    pdb_remove_ter_index,
    strip_hydrogens_keep_indices,
    write_xyz_frame,
)
from .tools_orca import (
    orca_calc_preset,
    orca_calculate_goat,
    orca_optimise_atoms,
    orca_preset_ccsd_gold,
    orca_preset_dft_cheap,
    orca_preset_dft_gold,
    orca_preset_mp2_gold,
    orca_preset_xtb,
)
from .tools_path import (
    cv_from_colvar,
    estimate_path_lambda,
    path_from_steered_md,
    select_frames_by_cv,
    select_frames_by_msd,
)
from .tools_plotting import (
    plot_images,
    plot_irc,
    plot_neb,
    plot_plumed,
    plot_plumed_multi,
    plot_temperature,
    plot_total_energy,
    show_atoms,
)
from .tools_plumed import (
    PLUMED_ASE_UNITS,
    find_molecules,
    plumed_calculator,
    plumed_metad_input,
    plumed_selection,
    run_opes_fes,
    run_sum_hills,
    sum_hills_files,
)
from .tools_reaction import (
    ConvergenceError,
    ConvergenceWarning,
    NebSummary,
    get_fmax,
    get_neb_path,
    get_ts_image,
    get_vibrations,
    optimise_geom,
    optimise_irc,
    optimise_neb,
    optimise_reactant_product,
    optimise_ts,
    prepare_neb,
    prepare_parallel_neb,
    quick_guess_path,
    quick_guess_ts,
    resample_path,
    restart_neb,
    restart_parallel_neb,
    socket_calculators,
    stitch_path,
    summarise_neb,
)
from .tools_style import ax_plot, n_plot
from .tools_units import (
    DEFAULT_ENERGY_UNIT,
    ENERGY_UNITS,
    as_kelvin,
    convert_energy,
    thermal_energy,
    unit_label,
)

__version__ = "0.2.0"

__all__ = [
    # tools_reaction
    "ConvergenceError",
    "ConvergenceWarning",
    "NebSummary",
    "summarise_neb",
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
    "orca_preset_dft_cheap",
    "orca_preset_dft_gold",
    "orca_preset_xtb",
    "orca_preset_mp2_gold",
    "orca_preset_ccsd_gold",
    # tools_path
    "cv_from_colvar",
    "estimate_path_lambda",
    "path_from_steered_md",
    "select_frames_by_cv",
    "select_frames_by_msd",
    # tools_geometry
    "bonded_cluster_indices_no_anchor_hub",
    "get_dimer_bonded_cluster_indices",
    "flip_and_face_bases",
    "optimize_with_fixed_anchors",
    "get_best_flip_and_face_bases",
    "swap_bonding_configuration",
    # tools_io
    "convert_pdb_to_xyz",
    "convert_xyz_to_pdb",
    "convert_xyz_to_plumed_ref",
    "element_from_pdb_line",
    "format_pdb_atom_name",
    "pdb_remove_ter_index",
    "strip_hydrogens_keep_indices",
    "write_xyz_frame",
    # tools_plumed
    "PLUMED_ASE_UNITS",
    "plumed_selection",
    "plumed_metad_input",
    "plumed_calculator",
    "find_molecules",
    "run_opes_fes",
    "run_sum_hills",
    "sum_hills_files",
    # tools_cv
    "as_positions",
    "plumed_angle_radians",
    "plumed_bias_and_fes",
    "plumed_input_1pt",
    "plumed_input_2pt_1d",
    "plumed_input_2pt_2d",
    "plumed_input_neb_path",
    "plumed_input_steered",
    "plumed_input_steered_pt",
    "plumed_one_based",
    "plumed_temperature_pair",
    "plumed_units_header",
    "switching_value",
    # tools_fes
    "FES",
    "FESSummary",
    "PlumedData",
    "as_fes",
    "fes_convergence",
    "fes_series_files",
    "load_fes_series",
    "summarise_fes",
    "plot_fes",
    "plot_fes_convergence",
    "plot_fes_1d",
    "plot_fes_2d",
    "plot_fes_2d_overlay",
    "plot_fes_slices",
    "plot_plumed_colvar",
    "plot_plumed_fes",
    "read_plumed_file",
    # tools_units
    "DEFAULT_ENERGY_UNIT",
    "ENERGY_UNITS",
    "as_kelvin",
    "convert_energy",
    "thermal_energy",
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
