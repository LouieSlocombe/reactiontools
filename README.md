# reactiontools

A centralised set of tools for looking at transition state (TS) and nudged
elastic band (NEB) calculations.

`reactiontools` wraps the parts of an [ASE](https://wiki.fysik.dtu.dk/ase/)
reaction-path workflow that get rewritten in every project: interpolating a
band, relaxing the endpoints, pulling the TS image out and refining it to a
true saddle point, following the IRC away from it, driving PLUMED, and
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

Runtime requirements are `numpy`, `scipy`, `matplotlib`, `pandas`, `ase>=3.23`
(the version where `NEB` moved to `ase.mep`) and
[`geodesic_interpolate`](https://github.com/LouieSlocombe/geodesic_interpolate)
(installed from git, used by `prepare_neb`, `quick_guess_path` and
`quick_guess_ts`).

Three dependencies are external — none is pulled in by `pip install`, and each
is only needed by the functions named:

| Dependency | Needed by | Notes |
| --- | --- | --- |
| [`sella`](https://github.com/zadorlab/sella) | `optimise_ts`, `optimise_irc` | Install with `pip install "reactiontools[ts]"`. Imported on first use, so the rest of the package works without it. |
| `plumed` executable | `run_sum_hills` | Must be on `PATH`. Called as a subprocess, not imported. |
| [ORCA](https://www.faccts.de/orca/) | everything in `tools_orca` | Licensed separately and installed by hand; point `ORCA_PATH` at the binary. See [build_tools/README.md](build_tools/README.md#orca). |

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

### Running the images in parallel

`prepare_neb` walks the band one image at a time, so a seven-image band costs
five sequential energy evaluations per step. That is fine for EMT and painful
for DFT. `prepare_parallel_neb` gives each interior image its own
[socket calculator](https://ase-lib.org/ase/calculators/socketio/socketio.html)
and asks ASE to spread the band over threads. Each thread blocks waiting on its
own socket while the external code works, which releases the GIL, so the
calculations genuinely overlap and a step costs about as much as its slowest
image:

```python
from ase.calculators.espresso import Espresso

from reactiontools import get_ts_image, optimise_neb, prepare_parallel_neb

def make_calc(index):
    # One directory per client: file-based codes write their input and output
    # relative to calc.directory, and clients sharing one overwrite each other.
    return Espresso(directory=f"image-{index}", pseudopotentials=...)

with prepare_parallel_neb(reactant, product, make_calc,
                          n_images=7, timeout=600) as neb:
    images = optimise_neb(neb, fmax=0.05, ts_traj="ts.traj")

# The sockets are shut by now, but the band read back from ts.traj carries its
# energies, so get_ts_image and plot_neb need no calculator.
ts = get_ts_image(images)
```

It is a context manager because the sockets and the client processes behind
them have to be shut down, including when the band blows up partway through.
Optimise inside the block; once it exits the calculators are closed.

Three things to know:

- **Run it as a single process.** The parallelism is threads and sockets, so
  give the ranks to the clients, not to the driver. Under `mpirun` ASE would
  distribute the images over MPI ranks instead and every rank would try to bind
  the same sockets, so this raises rather than hanging.
- **Only the interior images get sockets.** The endpoints are evaluated once
  and pinned, reusing the energy their calculator already holds — normally the
  one `optimise_reactant_product` left behind — and otherwise pricing them
  through the first socket. Two more clients idling all run for a pair of
  energies that are already known is not worth it.
- **Set a `timeout`.** Without one, a client that dies without closing its
  socket hangs the run forever rather than raising. `log="socket"` writes the
  i-PI traffic to `socket-0.log`, `socket-1.log`, … and is the first thing to
  reach for when a run stalls with no output.

Sockets are named `/tmp/ipi_reactiontools-<pid>-<image>` by default, so
concurrent jobs on one node do not collide. Pass `unixsocket="..."` to choose
the prefix, or `port=31415` to use TCP ports counting up from there instead.

To drive a Python calculator — an ML potential, say — in separate processes
rather than an external binary, pass `make_launcher` instead of `make_calc`:

```python
from ase.calculators.socketio import PySocketIOClient

with prepare_parallel_neb(reactant, product, None,
                          make_launcher=lambda index: PySocketIOClient(MyMLIP),
                          n_images=7, timeout=600) as neb:
    images = optimise_neb(neb, fmax=0.05)
```

`socket_calculators` is the same machinery without the band, for when you want
a pool of socket calculators for something else:

```python
from reactiontools import optimise_reactant_product, socket_calculators

with socket_calculators(1, make_calc) as (calc,):
    reactant, product = optimise_reactant_product(reactant, product, calc)
```

### Refining the transition state

A band gets close to the saddle but rarely converges tightly onto it, and the
top image is only ever as good as the spacing between images. `optimise_ts`
polishes it with [Sella](https://github.com/zadorlab/sella), `get_vibrations`
checks that the result really is a saddle, and `optimise_irc` confirms it
connects the two minima you meant:

```python
from reactiontools import (get_ts_image, get_vibrations, optimise_irc,
                           optimise_ts, plot_irc, stitch_path)

ts = optimise_ts(get_ts_image(images), calc, fmax=0.01)

# A minimum has all-real frequencies; a saddle has exactly one imaginary mode,
# which ASE reports as a complex number.
freqs = get_vibrations(ts, calc)
assert sum(f.imag != 0 for f in freqs) == 1

# Roll downhill both ways, then join the halves into one profile.
forward, reverse = optimise_irc(ts, calc, dx=0.1)
plot_irc(stitch_path(reverse, forward))
```

These two need Sella, which is an optional dependency — install it with
`pip install "reactiontools[ts]"`. They raise `ImportError` with that hint if
it is missing; nothing else in the package is affected.

### Building a flipped end state

A band needs a product, and for a stacked dimer that is the awkward structure
to draw by hand. `tools_geometry` builds one: work out which atoms make up each
half, then swap the halves over. The reflection that does the swap has a sign
convention that depends on how the fragments sit, so
`get_best_flip_and_face_bases` tries them all and keeps whichever leaves the
two centres of mass closest together:

```python
from reactiontools import (bonded_cluster_indices_no_anchor_hub,
                           get_best_flip_and_face_bases, prepare_neb)

anchors = [12, 37]  # one atom per half, where the two are joined
base_a = bonded_cluster_indices_no_anchor_hub(atoms, anchors[0])
base_b = bonded_cluster_indices_no_anchor_hub(atoms, anchors[1])

product = get_best_flip_and_face_bases(atoms, base_a, base_b, anchors,
                                       calc=calc)
neb = prepare_neb(atoms, product, calc, n_images=7)
```

Pass `optimise_after=False` to skip the relaxation and get the rigid swap
alone, in which case no calculator is needed.

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
| `prepare_parallel_neb(reactant, product, make_calc, n_images=5, ...)` | Context manager giving each interior image its own socket calculator, so the images are evaluated concurrently. |
| `socket_calculators(n_calculators, make_calc=None, ...)` | Context manager opening a pool of `SocketIOCalculator`s, one socket each, closed on exit. |
| `optimise_neb(neb, fmax=0.01, steps=1000, ts_traj='ts.traj')` | Relax the band and return the final images. |
| `get_ts_image(neb_images, calc=None)` | The highest-energy image along a band, reusing the energies the images carry unless a calculator is given. |
| `get_fmax(atoms)` | Largest per-atom force, the quantity the optimisers converge against. |
| `optimise_ts(ts_image, calc, fmax=0.01, eta=1e-4, gamma=0.1)` | Refine a TS guess to a true saddle point with Sella. Needs the `[ts]` extra. |
| `optimise_irc(ts_image, calc, dx=0.1, ...)` | Follow the IRC downhill in both directions, returning `(forward, reverse)`. Needs the `[ts]` extra. |
| `get_vibrations(atoms, calc)` | Finite-difference frequencies in cm⁻¹; one imaginary mode confirms a saddle. |
| `quick_guess_path(reactant, product, n_images=25)` | Geodesic path guess, no optimisation. |
| `quick_guess_ts(reactant, product, n_images=25)` | Midpoint of a geodesic guess, as a cheap TS starting structure. |

### `tools_orca` — ORCA calculators and conformer searches

Needs an ORCA install; see the dependency table above.

| Function | Description |
| --- | --- |
| `orca_calc_preset(calc_type='DFT', xc='r2SCAN-3c', charge=0, multiplicity=1, ...)` | Build an ASE ORCA calculator from presets for DFT, MP2, CCSD(T) or QM/XTB2, without hand-writing ORCA input. Drop the result into any function that takes a `calc`. |
| `orca_optimise_atoms(atoms, xc='r2SCAN-3c', tight_opt=True, ...)` | Relax a geometry with ORCA's own optimiser rather than ASE's, for molecules that BFGS in Cartesians struggles with. |
| `orca_calculate_goat(atoms, charge=0, multiplicity=1, n_procs=1)` | Run a GOAT conformer search, returning `(conformers, DataFrame)` of energies and populations. |

`f_solv` and `f_disp` take either `True` for the default (SMD water, D4) or a
string naming the solvent or dispersion keyword directly.

Worth running `orca_calculate_goat` before building a band: a NEB between two
arbitrary conformers explores the conformational change as well as the
reaction, and the barrier that comes back is not the one you wanted.

### `tools_geometry` — building product end states

A NEB needs a product as well as a reactant, and the product is usually the
awkward one to draw by hand. These build it: for a stacked dimer, find the two
halves and swap them over; for a proton transfer, move the hydrogen across.

| Function | Description |
| --- | --- |
| `bonded_cluster_indices_no_anchor_hub(atoms, anchor, mult=1.0, multi_h=1.3)` | Atoms bonded to an anchor, without the walk routing back through it. |
| `get_dimer_bonded_cluster_indices(atoms, anchors, mults=None, multi_h=1.3)` | Union of the two clusters, one per anchor. |
| `flip_and_face_bases(atoms, baseA_idxs, baseB_idxs, anchors, rot_matrix=None)` | Swap two fragments over, each landing on the other's anchor and facing it. |
| `optimize_with_fixed_anchors(atoms, baseA_idxs, baseB_idxs, anchor_indices, calc, fmax=0.05)` | Relax the fragments with their anchors pinned, leaving all other atoms untouched. |
| `get_best_flip_and_face_bases(atoms, baseA_idxs, baseB_idxs, anchors, optimise_after=True, calc=None)` | Search the reflection signs and keep whichever leaves the fragment centres of mass closest. |
| `swap_bonding_configuration(atoms, donor_index, hydrogen_index, acceptor_index)` | Turn O-H...O into O...H-O, keeping the O-H length, to make the product of a proton transfer. |

### `tools_plumed` — metadynamics support

| Function | Description |
| --- | --- |
| `plumed_selection(indices)` | Format zero-based atom indices as a one-based PLUMED selection with compact ranges. |
| `find_molecules(atoms)` | Split a structure into connected components using an ASE neighbour list. |
| `run_sum_hills(hills='HILLS', outfile='fes.dat', mintozero=True)` | Run `plumed sum_hills` to build a free-energy surface. |

### `tools_fes` — free-energy surfaces

Reads any PLUMED-style file — `COLVAR`, `fes.dat`, `HILLS`, `FES_from_State.py`
output — and plots it in one or two dimensions. Sources can be mixed freely:
paths, arrays, `(x, y, Z)` tuples or `FES` objects, one or many at a time, so a
single surface, a convergence series and a method comparison are all the same
call.

| Function | Description |
| --- | --- |
| `read_plumed_file(path, drop_der=True)` | Parse a PLUMED file into a `PlumedData` of columns, `#! FIELDS` names and `#! SET` metadata. |
| `as_fes(source, ...)` | Normalise any supported source into an `FES`. |
| `convert_energy(values, source, target)` | Convert between the units in `ENERGY_UNITS` (kJ/mol, kcal/mol, eV, meV, hartree, kT300). |
| `plot_fes(sources, **kwargs)` | Plot, dispatching on dimensionality. |
| `plot_fes_1d(sources, labels=None, energy_unit=None, ...)` | One or many 1-D profiles on one axes. |
| `plot_fes_2d(sources, levels=30, cmap=None, ...)` | Filled contours, one panel per surface. |
| `plot_fes_2d_overlay(sources, ...)` | Several 2-D surfaces as contour lines on shared axes. |
| `plot_fes_slices(sources, ...)` | 1-D cuts through a 2-D surface. |
| `plot_plumed_fes(path, ...)` | Convenience wrapper over `plot_fes` for a single file. |
| `plot_plumed_colvar(path, x_axis='time', columns=None, ...)` | One stacked panel per collective variable in a `COLVAR`. |

Energies are read as kJ/mol unless `source_unit` says otherwise, because that
is what PLUMED writes when driven from OpenMM. `max_energy` masks poorly
sampled regions rather than letting them dominate the colour scale, and
`filename=None` means write nothing.

### `tools_plotting` — figures

| Function | Description |
| --- | --- |
| `n_plot(xlab, ylab)` | Apply the house style to the current pyplot axes. |
| `ax_plot(fig, ax, xlab, ylab)` | Same, for an explicit `Figure`/`Axes` pair. `None` for either label leaves it untouched. |
| `plot_images(images, view='tilted', n_cols=4, ...)` | Grid of rendered structures, one panel per image. |
| `show_atoms(atoms, view='tilted', ...)` | Structures superimposed on one axes, for seeing how far a band has moved. |
| `plot_neb(images, calc=None, smooth=True, ...)` | NEB energy profile in meV against path length. |
| `plot_irc(images, calc=None, color='black', ...)` | The same profile with IRC defaults; pair with `stitch_path`. |
| `plot_temperature(trajectories, labels=None, timestep=None, ax=None)` | Temperature against frame or time for one or more trajectories. |
| `plot_total_energy(trajectories, labels=None, timestep=None, ax=None)` | Total energy against frame or time. |
| `plot_plumed(file='fes.dat', ...)` | One-dimensional PLUMED free-energy surface. |
| `plot_plumed_multi(files, mintozero=False, ...)` | Several surfaces overlaid; directories expand to the `fes.dat` files beneath them. |

`plot_plumed` and `plot_plumed_multi` are ASE-flavoured wrappers over
`tools_fes`: they assume `fes.dat` is in eV, as `plumed sum_hills` writes it
for an ASE-driven run, and plot meV. For a run driven from OpenMM the file is
in kJ/mol — use `tools_fes` directly and set `source_unit`.

Plotting functions return `(fig, ax)` so you can keep customising, and accept an
existing `fig`/`ax` to compose subplots. `plot_images` returns `(fig, axes)`
with a flat array of panels. Those that take `save=True` write both `.png` (600
dpi) and `.pdf`.

`plot_images` accepts the named views `"top"`, `"side"`, `"front"` and
`"tilted"`, or any raw ASE rotation string such as `"300x,0y,0z"`.

### Units

ASE works in eV and Å, and that is what the functions take and return. The
plotting layer converts to meV for readability: `plot_neb` shifts energies so
the lowest image sits at zero, and `plot_plumed`/`plot_plumed_multi` scale
`fes.dat` by 1000.

`tools_fes` is the exception, because PLUMED's units depend on what drove it:
it defaults to kJ/mol, and any of the units in `ENERGY_UNITS` can be selected
per call with `source_unit` and `energy_unit`.

## Testing

```bash
pytest
```

The suite builds its own structures with `ase.build` and evaluates them with
EMT, so it needs no external data or calculator.

## Citing

If `reactiontools` is useful in your work, please cite it and whichever of
the codes it wraps you actually exercised — all in
[CITATIONS.bib](CITATIONS.bib):

| Entry | Cite for | Used by |
| --- | --- | --- |
| `Slocombe_reactiontools` | `reactiontools` itself | always |
| `larsen2017atomic` | [ASE](https://wiki.fysik.dtu.dk/ase/) | NEB, optimisation and I/O throughout |
| `zhu2019geodesic` | [`geodesic_interpolate`](https://github.com/LouieSlocombe/geodesic_interpolate) | `prepare_neb(geo_int=True)`, `quick_guess_path`, `quick_guess_ts` |
| `hermes2022sella` | [Sella](https://github.com/zadorlab/sella) | `optimise_ts`, `optimise_irc` |
| `plumed2` | [PLUMED](https://www.plumed.org/) | `run_sum_hills` |
| `jonsson1998nudged`, `henkelman2000improved`, `henkelman2000climbing` | The NEB method, the improved tangent and the climbing image | `prepare_neb`, `optimise_neb` |
| `smidstrup2014improved` | IDPP interpolation | `prepare_neb(geo_int=False)` |
| `nocedal2006numerical` | BFGS | every `optimise_*` that is not Sella |
| `neese2012orca`, `neese2022orca5`, `neese2025orca6` | [ORCA](https://www.faccts.de/orca/) | everything in `tools_orca` |
| `desouza2025goat` | The GOAT conformer search | `orca_calculate_goat` |
| `grimme2021r2scan3c`, `furness2020r2scan` | The default `r2SCAN-3c` functional | `orca_calc_preset`, `orca_optimise_atoms` |
| `caldeweyher2019d4` | D4 dispersion | `f_disp=True` |
| `barone1998cpcm`, `marenich2009smd` | CPCM/SMD implicit solvation | `f_solv` |
| `riplinger2013efficient`, `riplinger2013natural`, `pinski2015sparse` | DLPNO-MP2 and DLPNO-CCSD(T) | `calc_type='MP2'`, `calc_type='CCSD'` |
| `bannwarth2019gfn2` | GFN2-xTB | `calc_type='QM/XTB2'` |

## License

MIT — see [LICENSE](LICENSE).
