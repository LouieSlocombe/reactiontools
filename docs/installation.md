# Installation

From PyPI, into an existing environment:

```bash
pip install reactiontools
```

That is enough for everything except the saddle-point searches, which need
Sella, and anything in `tools_orca`, which needs ORCA. Both are installed by
hand — see [Dependencies](#dependencies). Geodesic interpolation needs nothing
extra.

Sella is one more command, and worth running now if you expect to use
`optimise_ts`, `optimise_irc` or `sella_ts_search`:

```bash
pip install git+https://github.com/LouieSlocombe/sella.git
```

For everything at once — environment, PLUMED and the package — one command from
the repository root:

```bash
bash build_tools/conda_install.sh
```

That creates the `reactiontools` conda environment, compiles PLUMED with the
OPES module and the matching Python bindings into it, and installs this package
— along with `sella`, cloned next to this repository — in editable mode. It
**removes and recreates** any environment of that name; pass `ENV_NAME=...` to
install somewhere else, or `SRC_DIR=...` to keep the checkouts elsewhere. Checkouts that already exist are used as they are and never
wiped.

If you already have a PLUMED with OPES on your `PATH`, the environment and the
package on their own are:

```bash
conda env create -f build_tools/environment.yml -y
```

```bash
conda activate reactiontools
```

```bash
pip install -e .
```

See [build_tools/README.md](https://github.com/LouieSlocombe/reactiontools/blob/main/build_tools/README.md) for the full guide, why
PLUMED is built from source, and the Sol cluster route.

## Dependencies

Python 3.12 or newer is required, and `pip install reactiontools` brings in
`numpy>=2.0`, `scipy>=1.16`, `matplotlib>=3.8.4`, `pandas>=2.2.2`, `ase>=3.25`
and `mdtraj>=1.10.2`.

The Python, ASE and SciPy minimums follow what `tools_geodesic` needs.
NumPy 2 provides the trapezoidal integration used for basin free energies;
the Matplotlib, pandas and MDTraj minimums support NumPy 2.

### Geodesic interpolation

Nothing to install. `prepare_neb(geo_int=True)`, `quick_guess_path`,
`quick_guess_ts` and `seed_product_from_ts` are built on `tools_geodesic`,
which is part of the package like any other module. `geodesic_interpolate` is
its entry point, and `Geodesic`, `redistribute` and the scalers that set the
metric are exported alongside it.

It is derived from
[`geodesic-interpolate`](https://github.com/virtualzx-nad/geodesic-interpolate)
by Xiaolei Zhu, MIT licensed as the rest of the package is — the notice is at
the foot of `LICENSE` — and it is still `zhu2019geodesic` you cite when you use
it. It lives here rather than being depended on because the version on PyPI
under the name `geodesic-interpolate` exposes only the lower-level `Geodesic`
and `redistribute`, without the ASE-aware entry point these functions call, and
a git dependency on a fork that does have it would keep this package off PyPI
altogether.

### Sella

[`sella`](https://github.com/LouieSlocombe/sella) is **not** installed with the
package, for the same PyPI reason: the fork these workflows are built against is
not on PyPI. Install it by hand:

```bash
pip install git+https://github.com/LouieSlocombe/sella.git
```

Only `optimise_ts`, `optimise_irc` and `sella_ts_search` need it, and they are
the only things that break without it — each raises an `ImportError` repeating
the command above. Everything else, the NEB and PLUMED halves of the package
included, works without it.

### Development

The test runner and the linter are not runtime dependencies either. To run the
suite from a checkout, install the `dev` extra:

```bash
pip install -e ".[dev]"
```

### Outside pip entirely

Three more fall outside `pip` altogether, and are only needed by the functions
named:

| Dependency | Needed by | Notes |
| --- | --- | --- |
| `plumed` executable | `run_sum_hills` | Must be on `PATH`. Called as a subprocess, not imported. Compiled by `conda_install.sh`; conda-forge's `plumed` package is built without the OPES module that `f_opes=True` inputs need. |
| `py-plumed` | `plumed_calculator` | The Python bindings, compiled by `conda_install.sh` against the same PLUMED. Imported on first use; the input builder works without it. |
| [ORCA](https://www.faccts.de/orca/) | everything in `tools_orca` | Licensed separately and installed by hand; point `ORCA_PATH` at the binary. See [build_tools/README.md](https://github.com/LouieSlocombe/reactiontools/blob/main/build_tools/README.md#orca). |
