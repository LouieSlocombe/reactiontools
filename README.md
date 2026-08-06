# reactiontools

A centralised set of tools for looking at transition state (TS) and nudged
elastic band (NEB) calculations.

`reactiontools` wraps the parts of an [ASE](https://wiki.fysik.dtu.dk/ase/)
reaction-path workflow that get rewritten in every project: interpolating a
band, relaxing the endpoints, pulling the TS image out, driving PLUMED, and
producing publication-ready figures with consistent styling.

It is calculator-agnostic — anything that behaves like an ASE calculator works,
from EMT to a machine-learned potential to a DFT code.

## Installation

Create the conda environment and install in editable mode:

```bash
conda env create -f build_tools/environment.yml -y
```

```bash
conda activate reactiontools
```

```bash
pip install -e ".[dev]"
```

See [build_tools/README.md](build_tools/README.md) for the full guide,
including building PLUMED with the OPES module.

### Dependencies

Runtime requirements are `numpy`, `scipy`, `matplotlib`, `ase>=3.23` (the
version where `NEB` moved to `ase.mep`) and
[`geodesic_interpolate`](https://github.com/LouieSlocombe/geodesic_interpolate)
(installed from git, used by `prepare_neb`, `quick_guess_path` and
`quick_guess_ts`).

One dependency is optional:

| Dependency | Needed by | Notes |
| --- | --- | --- |
| `plumed` executable | `run_sum_hills` | Must be on `PATH`. Called as a subprocess, not imported. |

## Quickstart

Relax two endpoints, run a climbing-image NEB between them, and plot the
barrier. This example is a gold adatom hopping between hollow sites on an
Al(100) surface, and runs in a few seconds with EMT:

```python
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from reactiontools import (get_ts_image, optimise_neb,
                           optimise_reactant_product, plot_neb, prepare_neb)

calc = EMT()

slab = fcc100("Al", size=(2, 2, 3))
add_adsorbate(slab, "Au", 1.7, "hollow")
slab.center(axis=2, vacuum=4.0)
slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))

reactant = slab.copy()
product = slab.copy()
product.positions[-1, 0] += product.cell[0, 0] / 2  # hop to the next site

reactant, product = optimise_reactant_product(reactant, product, calc, fmax=0.05)

# rm_ro_trans=False: the slab is periodic and constrained, so the rigid-body
# degrees of freedom are already pinned. See "Choosing NEB settings" below.
neb = prepare_neb(reactant, product, calc, n_images=7,
                  climb=True, rm_ro_trans=False, geo_int=False)
images = optimise_neb(neb, fmax=0.05, ts_traj="ts.traj")

ts = get_ts_image(images, calc)
print(f"Barrier: {ts.get_potential_energy() - reactant.get_potential_energy():.3f} eV")

plot_neb(images, calc, filename="neb")
```

This converges in 17 NEB steps and prints `Barrier: 0.374 eV`. `plot_neb`
writes `neb.png` and `neb.pdf`, with energies referenced to the lowest image
and reported in meV.

### Choosing NEB settings

`prepare_neb`'s defaults suit isolated molecules. A few are worth thinking
about:

- **`rm_ro_trans`** removes rigid-body rotation and translation. That is right
  for a molecule tumbling in vacuum, but wrong for a periodic slab whose atoms
  are already pinned by a constraint — leaving it on stops the band converging.
  Set `rm_ro_trans=False` for constrained or periodic systems.
- **`geo_int`** uses geodesic interpolation. With `geo_int=False` the band is
  built by linear interpolation refined with IDPP, which is usually fine for
  a small displacement like the hop above.
- **`parallel`** evaluates the interior images concurrently instead of one at
  a time, for both the initial energies and every force call `optimise_neb`
  makes afterwards. Without an MPI launcher this runs each image's calculator
  in its own thread, which only helps if `calc` releases the GIL while it
  runs (e.g. one that shells out to an external code — a Python-only
  calculator like EMT gains nothing). Run under `mpirun` and ASE instead
  distributes the images across MPI ranks; pass a specific communicator with
  `world` if you don't want `ase.parallel.world`.

### Minimising with a socket calculator

`optimise_geom` and `optimise_reactant_product` take the same `use_socket`,
`socket_port`, `socket_unixsocket` and `socket_log` arguments. With
`use_socket=True`, `calc` is driven through an ASE
[`SocketIOCalculator`](https://wiki.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html)
instead of being called directly, so the external program launches once and
stays running for every BFGS step instead of restarting on each one:

```python
reactant, product = optimise_reactant_product(
    reactant, product, calc, fmax=0.05, use_socket=True)
```

This needs a calculator ASE knows how to launch as an i-PI client — built-in
support covers `Espresso`, `Aims` and `Siesta`. A calculator without that
support, such as EMT, will fail with `use_socket=True`.

Already have a band on disk? Read it back and plot it directly:

```python
from ase.calculators.emt import EMT
from ase.io import read
from reactiontools import plot_images, plot_neb

images = read("ts.traj", index="-7:")  # the last band written by optimise_neb
plot_images(images, view="side", n_cols=4, save=True)
plot_neb(images, EMT(), smooth=True)
```

Images read back from a trajectory already carry their energies, so `plot_neb`
reuses them and only falls back to the calculator you pass for images that have
none.

### PLUMED

Build a selection string, sum the hills, and plot the free-energy surface:

```python
from ase.io import read
from reactiontools import find_molecules, plumed_selection, run_sum_hills, plot_plumed

atoms = read("system.xyz")
solute = find_molecules(atoms)[0]
print(plumed_selection(solute))  # e.g. "1-9,14" — one-based, as PLUMED expects

run_sum_hills(hills="HILLS", outfile="fes.dat", mintozero=True)
plot_plumed("fes.dat", x_label="CV (Å)")
```

Comparing several runs laid out as `<run>/fes.dat` — pass the parent directory
and every surface beneath it is found and labelled by run name:

```python
from reactiontools import plot_plumed_multi

plot_plumed_multi("runs/", mintozero=True, x_label="CV (Å)")
```

## API reference

### `tools_reaction` — building and analysing paths

| Function | Description |
| --- | --- |
| `get_neb_path(images)` | Cumulative reaction-path distance along a band, starting at zero. |
| `stitch_path(path1, path2, f_reverse_path=False)` | Join a reactant-side and product-side path into one IRC-like sequence. |
| `resample_path(path, n_resample)` | Cubic-spline resample a path to a fixed number of images, preserving the endpoints. |
| `optimise_geom(atoms, calc, ..., use_socket=False)` | Relax a structure with BFGS and return the final image. `use_socket=True` drives `calc` over an ASE `SocketIOCalculator`. |
| `optimise_reactant_product(reactant, product, calc, ..., use_socket=False)` | Relax both endpoints independently, one after the other. |
| `prepare_neb(reactant, product, calc, n_images=5, climb=True, geo_int=True, k=2.0, parallel=False)` | Build a configured `ase.mep.NEB`, interpolating geodesically or with IDPP. `parallel=True` evaluates images concurrently. |
| `optimise_neb(neb, fmax=0.01, steps=1000, ts_traj='ts.traj')` | Relax the band and return the final images. |
| `get_ts_image(neb_images, calc)` | The highest-energy image along a band. |
| `quick_guess_path(reactant, product, n_images=25)` | Geodesic path guess, no optimisation. |
| `quick_guess_ts(reactant, product, n_images=25)` | Midpoint of a geodesic guess, as a cheap TS starting structure. |

### `tools_plumed` — metadynamics support

| Function | Description |
| --- | --- |
| `plumed_selection(indices)` | Format zero-based atom indices as a one-based PLUMED selection with compact ranges. |
| `find_molecules(atoms)` | Split a structure into connected components using an ASE neighbour list. |
| `run_sum_hills(hills='HILLS', outfile='fes.dat', mintozero=True)` | Run `plumed sum_hills` to build a free-energy surface. |

### `tools_plotting` — figures

| Function | Description |
| --- | --- |
| `n_plot(xlab, ylab)` | Apply the house style to the current pyplot axes. |
| `ax_plot(fig, ax, xlab, ylab)` | Same, for an explicit `Figure`/`Axes` pair. |
| `plot_images(images, view='tilted', n_cols=4, ...)` | Grid of rendered structures, one panel per image. |
| `plot_neb(images, calc, smooth=True, ...)` | NEB energy profile in meV against path length. |
| `plot_temperature(trajectories, labels=None, timestep=None, ax=None)` | Temperature against frame or time for one or more trajectories. |
| `plot_total_energy(trajectories, labels=None, timestep=None, ax=None)` | Total energy against frame or time. |
| `plot_plumed(file='fes.dat', ...)` | One-dimensional PLUMED free-energy surface. |
| `plot_plumed_multi(files, mintozero=False, ...)` | Several surfaces overlaid; directories expand to the `fes.dat` files beneath them. |

Plotting functions return `(fig, ax)` so you can keep customising, and accept an
existing `fig`/`ax` to compose subplots. `plot_images` returns `(fig, axes)`
with a flat array of panels. Those that take `save=True` write both `.png` (600
dpi) and `.pdf`.

`plot_images` accepts the named views `"top"`, `"side"`, `"front"` and
`"tilted"`, or any raw ASE rotation string such as `"300x,0y,0z"`.

### Units

ASE works in eV and Å, and that is what the functions take and return. The
plotting layer converts to meV for readability: `plot_neb` shifts energies so
the lowest image sits at zero, and the PLUMED readers scale `fes.dat` by 1000.

## Testing

```bash
pytest
```

The suite builds its own structures with `ase.build` and evaluates them with
EMT, so it needs no external data or calculator.

## License

MIT — see [LICENSE](LICENSE).
