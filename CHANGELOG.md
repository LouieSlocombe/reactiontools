# Changelog

All notable changes to `reactiontools` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html): within a
major version, anything exported from the top-level `reactiontools` namespace
keeps its name and its meaning.

## [Unreleased]

## [1.0.0] — 2026-09-18

First public release. Versions before this one were developed in the open but
never tagged, so everything the package does arrives here at once.

### Added

- `tools_reaction` — build, relax and post-process nudged elastic bands:
  `prepare_neb` / `optimise_neb` with IDPP or geodesic interpolation and the
  climbing image, `restart_neb` to continue a band that has already been
  relaxed, `prepare_parallel_neb` and `prepare_threaded_neb` to spread the
  images over socket or file-based calculators, `optimise_ts` to refine the top
  of a band into a true saddle point with Sella, `optimise_irc` to follow the
  intrinsic reaction coordinate away from it, and `summarise_neb` for the
  forward barrier, reverse barrier and reaction energy.
- `tools_geodesic` — the interpolation `prepare_neb(geo_int=True)`,
  `quick_guess_path` and `quick_guess_ts` are built on. `geodesic_interpolate`
  takes two end states, or more, to a path that is short in a metric of scaled
  inter-atomic distances rather than in Cartesian space, so atoms do not run
  through each other on the way across; `redistribute` and `Geodesic` are the
  two stages it runs, and the scalers that set the metric are exported too.
- `tools_geometry` — build end states: `swap_bonding_configuration` to move
  protons across their hydrogen bonds, `flip_and_face_bases` for stacked
  dimers, and `seed_product_from_ts` / `seed_minima_from_ts` to find the minima
  either side of a saddle point. Also `kabsch_transform`, `align_atom_sets` and
  `atom_set_rmsd` for superposing two structures.
- `tools_orca` — ORCA calculators from named presets, from xTB and "3c"
  screening up to compound CCSD(T)/CBS `orca_gold_standard` energies, plus
  `orca_optimise_atoms`, the `orca_calculate_goat` conformer search and a
  Sella saddle search in `sella_ts_search`.
- `tools_cv`, `tools_path`, `tools_plumed` — collective variables for proton
  transfer and for progress along a reference path, the PLUMED input that
  biases them, `path_from_steered_md` to turn a steered trajectory into a
  `PATHMSD` reference, and `plumed_calculator` to bias an ASE dynamics run.
- `tools_fes` — read `COLVAR`, `HILLS` and `fes.dat`, measure barriers and
  basin free-energy differences with `summarise_fes`, and track those numbers
  across a series of surfaces with `fes_convergence` to judge convergence.
  Surfaces from an `OPES_METAD` run, which deposits no hills, are read back out
  of the `STATE` file.
- `tools_io`, `tools_units`, `tools_plotting`, `tools_style` — structure-file
  conversion, the energy units everything else converts between, and
  publication-ready figures with consistent styling.
- Convergence reporting across every `optimise_*` function: whether the force
  criterion was actually reached is recorded in `info["converged"]` on the
  structures returned, and a `ConvergenceWarning` is issued when it was not.
  Pass `raise_on_unconverged=True` for a `ConvergenceError` instead.
- Type annotations throughout the public API.
- Documentation at [reactiontools.readthedocs.io](https://reactiontools.readthedocs.io),
  covering installation, a quickstart, guides for each workflow and a generated
  API reference.
- `build_tools/conda_install.sh`, which creates the environment, compiles
  PLUMED with the OPES module and its Python bindings, and installs this
  package and its two git dependencies in editable mode.

### Fixed

- `seed_product_from_ts` decided whether the seed had moved anywhere by
  comparing two RMSDs exactly. When the push covers no ground the seed *is* the
  transition state, so those are one number computed two ways, and which came
  out larger was decided by rounding in the Kabsch alignment — about 1e-16 Å,
  either sign depending on the LAPACK underneath. On an unlucky build the
  function stayed silent and reported `info["seeded"] = True` for a seed that
  had not moved at all. Both the warning and the flag now allow a 1e-9 Å
  tolerance, and stay exact complements of each other.

### Notes

- `pytest`, `pytest-cov` and `ruff` moved out of the runtime dependencies into
  a `dev` extra; installing the package no longer pulls in a test runner and a
  linter. Install them with `pip install -e ".[dev]"`.

### Dependencies

Neither of the two dependencies that used to come from git is declared as such
any more. PyPI rejects any distribution whose metadata carries a direct URL, so
this is what makes `pip install reactiontools` possible at all.

- **geodesic_interpolate is no longer a dependency.** The interpolation is part
  of the package, as `tools_geodesic`, so `prepare_neb(geo_int=True)`,
  `quick_guess_path`, `quick_guess_ts` and `seed_product_from_ts` need nothing
  installed alongside. It is derived from
  [`geodesic-interpolate`](https://github.com/virtualzx-nad/geodesic-interpolate),
  MIT licensed and copyright Xiaolei Zhu; that notice is at the foot of
  `LICENSE` and ships with every copy of this package. The distribution
  published on PyPI under the name `geodesic-interpolate` was not usable in its
  place: it exposes only the lower-level `Geodesic` and `redistribute`, without
  the ASE-aware entry point these functions call. Keep citing `zhu2019geodesic`
  for it.
- **sella is no longer installed with the package.** Only `optimise_ts`,
  `optimise_irc` and `sella_ts_search` use it, they import it on demand, and
  each raises an `ImportError` carrying the install command when it is missing:
  `pip install git+https://github.com/LouieSlocombe/sella.git`. Everything else
  works without it.

[Unreleased]: https://github.com/LouieSlocombe/reactiontools/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LouieSlocombe/reactiontools/releases/tag/v1.0.0
