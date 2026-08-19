# reactiontools

A centralised set of tools for looking at transition state (TS) and nudged
elastic band (NEB) calculations.

`reactiontools` wraps the parts of an [ASE](https://wiki.fysik.dtu.dk/ase/)
reaction-path workflow that get rewritten in every project: interpolating a
band, relaxing the endpoints, pulling the TS image out and refining it to a
true saddle point, following the IRC away from it, driving PLUMED, and
producing publication-ready figures with consistent styling.

It also holds the collective variables an enhanced-sampling run is biased
along — `tools_cv` for proton transfer, `tools_path` for turning a steered
trajectory into a `PATHMSD` reference. Those are written
as text for whatever runs PLUMED, so they are equally usable from an ASE run
here or from an OpenMM one driven by
[openmmnqe](https://github.com/LouieSlocombe/openmmnqe), which depends on this
package for exactly that.

It is calculator-agnostic — anything that behaves like an ASE calculator works,
from EMT to a machine-learned potential to a DFT code.

## Installation

One command, from the repository root:

```bash
bash build_tools/conda_install.sh
```

That creates the `reactiontools` conda environment, compiles PLUMED with the
OPES module and the matching Python bindings into it, and installs this package
— along with `geodesic_interpolate` and `sella`, cloned next to this repository —
in editable mode. It **removes and recreates** any environment of that name;
pass `ENV_NAME=...` to install somewhere else, or `SRC_DIR=...` to keep the
checkouts elsewhere. Checkouts that already exist are used as they are and never
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

See [build_tools/README.md](build_tools/README.md) for the full guide, why
PLUMED is built from source, and the Sol cluster route.

### Dependencies

Installed requirements include `numpy`, `scipy`, `matplotlib`, `pandas`,
`ase>=3.23` (the version where `NEB` moved to `ase.mep`), `mdtraj`, pytest and
Ruff, plus two that come from git:
[`sella`](https://github.com/LouieSlocombe/sella) (saddle-point refinement and
IRC) and
[`geodesic_interpolate`](https://github.com/LouieSlocombe/geodesic_interpolate)
(used by `prepare_neb`, `quick_guess_path` and `quick_guess_ts`).
`conda_install.sh` sets both up as editable checkouts beside this repository;
a plain `pip install` takes them from GitHub instead.

Three dependencies fall outside `pip install` and are only needed by the
functions named:

| Dependency | Needed by | Notes |
| --- | --- | --- |
| `plumed` executable | `run_sum_hills` | Must be on `PATH`. Called as a subprocess, not imported. Compiled by `conda_install.sh`; conda-forge's `plumed` package is built without the OPES module that `f_opes=True` inputs need. |
| `py-plumed` | `plumed_calculator` | The Python bindings, compiled by `conda_install.sh` against the same PLUMED. Imported on first use; the input builder works without it. |
| [ORCA](https://www.faccts.de/orca/) | everything in `tools_orca` | Licensed separately and installed by hand; point `ORCA_PATH` at the binary. See [build_tools/README.md](build_tools/README.md#orca). |

## Quickstart

Relax two endpoints, run a climbing-image NEB between them, and plot the
barrier. This example is a gold adatom hopping between hollow sites on an
Al(100) surface, and runs in a few seconds with EMT:

```python
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from reactiontools import (
    optimise_neb,
    optimise_reactant_product,
    plot_neb,
    prepare_neb,
    summarise_neb,
)

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
neb = prepare_neb(
    reactant, product, calc, n_images=7, climb=True, rm_ro_trans=False, geo_int=False
)
images = optimise_neb(neb, fmax=0.05, ts_traj="ts.traj")

print(summarise_neb(images))

plot_neb(images, calc, filename="neb")
```

This converges in 17 NEB steps and prints:

```
Barrier:         0.374 eV
Reverse barrier: 0.374 eV
Reaction energy: 0.000 eV
TS image:        3 of 6
```

`plot_neb` writes `neb.png` and `neb.pdf`, with energies referenced to the
lowest image and reported in meV.

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

### Reading the numbers off a band

`summarise_neb` reduces a relaxed band to what it was run for:

```python
summary = summarise_neb(images)
summary.barrier  # forward, eV
summary.reverse_barrier  # eV
summary.reaction_energy  # product - reactant, eV; negative if exothermic
summary.ts_index  # which image get_ts_image returns
summary.energies  # absolute, per image, eV
```

The barrier is measured from the highest image, so it agrees with
`get_ts_image` and with the profile `plot_neb` draws. It is not spline-fitted,
unlike ASE's `NEBTools.get_barrier`, whose default interpolates between images
and can report a maximum sitting at no image at all.

`summary.is_barrierless` is worth checking before paying for a saddle search:
it is `True` when the top image is an endpoint, meaning the path runs downhill
throughout and `get_ts_image` would hand `optimise_ts` a structure that is not
a saddle.

```python
summary = summarise_neb(images)
if not summary.is_barrierless:
    ts = optimise_ts(get_ts_image(images), calc, fmax=0.01)
```

A band resolves the barrier only as well as its images allow — the true saddle
lies between them, so this underestimates. Refining the top image is what turns
it into a number worth quoting.

`plot_neb` can put the same figure on the plot, in the meV of its y-axis. It is
off by default, so existing figures do not change:

```python
plot_neb(images, calc, annotate=True)
```

### Knowing whether it converged

Every `optimise_*` function records whether it actually reached `fmax` in
`info["converged"]` on the structures it hands back, and warns
`ConvergenceWarning` when it did not. A run that quietly hits its step limit is
the expensive kind of mistake: nothing looks wrong until the vibrational
analysis, several jobs later.

```python
images = optimise_neb(neb, fmax=0.05, steps=200)
if not images[0].info["converged"]:
    print("band still moving — restart from ts.traj with more steps")
```

For a script that should stop rather than carry on with a half-relaxed
structure, ask for an exception instead:

```python
reactant, product = optimise_reactant_product(
    reactant, product, calc, fmax=0.05, raise_on_unconverged=True
)
```

`raise_on_unconverged` is per call. To hold a whole script to it, promote the
warning once:

```python
import warnings
from reactiontools import ConvergenceWarning

warnings.simplefilter("error", ConvergenceWarning)
```

The flag never costs you the work already done: `optimise_neb` writes
`ts_traj` before the check, and `optimise_irc` runs both directions before
reporting either, so the trajectories are on disk to restart from even when the
call raises.

### Choosing the optimiser, and where it logs

Every function that relaxes something takes an `optimiser`, defaulting to
BFGS. Pass any ASE optimiser class — FIRE is the usual second thing to try on
a band BFGS cannot settle, being less easily thrown by the spring forces:

```python
from ase.optimize import FIRE

images = optimise_neb(neb, fmax=0.05, optimiser=FIRE)
```

Anything callable as `optimiser(atoms, trajectory=..., logfile=...)` works, so
`functools.partial` is how an optimiser's own settings get through:

```python
from functools import partial

reactant = optimise_geom(reactant, calc, optimiser=partial(FIRE, a=0.15))
```

`optimise_ts` and `optimise_irc` have no `optimiser` argument — the search
there is Sella's, which is the point of them.

`logfile` says where the per-step table goes, following ASE's convention:
`'-'` is stdout and the default, a filename writes there instead, and `None`
silences it. Useful when a band's log would otherwise bury everything else:

```python
images = optimise_neb(neb, fmax=0.05, logfile="neb.log")
```

This covers the optimiser's own output. The few progress lines the package
prints itself — `Optimising reactant...`, and the energy and force `optimise_ts`
reports before it starts — still go to stdout.

`optimise_geom` deletes its trajectory once it has read the final structure
back, since a successful relaxation needs nothing else from it. When one
misbehaves, the path it took is the evidence, so keep it:

```python
relaxed = optimise_geom(atoms, calc, opti_traj="opt.traj", keep_traj=True)
```

That also holds when the run raises: with `keep_traj=True` the trajectory
survives a `ConvergenceError`, which is exactly the case worth looking at.

### Continuing a band

A band that ran out of steps is not wasted work — it is a better starting path
than any interpolation. `restart_neb` takes the images back and builds a fresh
NEB around them:

```python
from reactiontools import optimise_neb, restart_neb

images = optimise_neb(neb, fmax=0.05, steps=200)
if not images[0].info["converged"]:
    images = optimise_neb(restart_neb(images, calc), fmax=0.05, steps=500)
```

The same call covers the other reasons to go round again: tighten `fmax`, swap
in a better calculator, or turn on climbing for a second pass having left it
off for the first, which is the usual way to run a band that is expensive to
converge.

```python
neb = restart_neb(images, better_calc, climb=True)
```

From disk it is the last `n_images` entries of the trajectory, since the
optimiser writes the whole band on every step:

```python
from ase.io import read

images = read("ts.traj", index="-7:")
neb = restart_neb(images, calc)
```

Pass `n_images` to resample the band on the way in — for a path too coarse to
resolve the barrier, or one whose images have bunched up, since resampling
spaces them evenly along the path:

```python
neb = restart_neb(images, calc, n_images=11)
```

`restart_parallel_neb` is the same thing over sockets, and is a context manager
like `prepare_parallel_neb`. It costs less than starting a parallel band from
scratch: a band read back from a trajectory carries its endpoint energies, so
those are reused rather than priced through a socket.

```python
with restart_parallel_neb(images, make_calc, timeout=600) as neb:
    images = optimise_neb(neb, fmax=0.05, steps=500)
```

Both copy the images they are given, so the band you passed in stays as it was
— it is what to fall back on if the restart goes worse than the run it
continues. Neither interpolates, so there is no `geo_int` argument.

**Give the restart the same band settings as the run it continues.** A band
records its geometries and nothing else, so `rm_ro_trans`, `k` and `climb` all
have to be supplied again. `rm_ro_trans` is the one that bites: it defaults to
`True`, as in `prepare_neb`, and leaving it there for the Al(100) slab from the
quickstart — built with `rm_ro_trans=False`, being periodic and constrained —
stops the continued band converging just as surely as it would have stopped the
first one. Continuing that quickstart properly means:

```python
neb = restart_neb(images, calc, climb=True, rm_ro_trans=False)
```

### Minimising with a socket calculator

`optimise_geom` and `optimise_reactant_product` take the same `use_socket`,
`socket_port`, `socket_unixsocket` and `socket_log` arguments. With
`use_socket=True`, `calc` is driven through an ASE
[`SocketIOCalculator`](https://wiki.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html)
instead of being called directly, so the external program launches once and
stays running for every BFGS step instead of restarting on each one:

```python
reactant, product = optimise_reactant_product(
    reactant, product, calc, fmax=0.05, use_socket=True
)
```

This needs a calculator ASE knows how to launch as an i-PI client — built-in
support covers `Espresso`, `Aims` and `Siesta`. A calculator without that
support, such as EMT, will fail with `use_socket=True`.

Already have a band on disk? Read it back and plot it directly — or hand it to
`restart_neb` and carry on relaxing it:

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
none. One thing they do not carry is `info["converged"]`, which is written onto
the images `optimise_neb` returns rather than into the trajectory itself.

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


with prepare_parallel_neb(reactant, product, make_calc, n_images=7, timeout=600) as neb:
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

with prepare_parallel_neb(
    reactant,
    product,
    None,
    make_launcher=lambda index: PySocketIOClient(MyMLIP),
    n_images=7,
    timeout=600,
) as neb:
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
from reactiontools import (
    get_ts_image,
    get_vibrations,
    optimise_irc,
    optimise_ts,
    plot_irc,
    stitch_path,
)

ts = optimise_ts(get_ts_image(images), calc, fmax=0.01)

# A minimum has all-real frequencies; a saddle has exactly one imaginary mode,
# which ASE reports as a complex number.
freqs = get_vibrations(ts, calc)
assert sum(f.imag != 0 for f in freqs) == 1

# Roll downhill both ways, then join the halves into one profile.
forward, reverse = optimise_irc(ts, calc, dx=0.1)
plot_irc(stitch_path(reverse, forward))
```

These two use Sella, which is installed with `reactiontools` and imported only
when a saddle-point calculation is requested.

### Building a flipped end state

A band needs a product, and for a stacked dimer that is the awkward structure
to draw by hand. `tools_geometry` builds one: work out which atoms make up each
half, then swap the halves over. The reflection that does the swap has a sign
convention that depends on how the fragments sit, so
`get_best_flip_and_face_bases` tries them all and keeps whichever leaves the
two centres of mass closest together:

```python
from reactiontools import (
    bonded_cluster_indices_no_anchor_hub,
    get_best_flip_and_face_bases,
    prepare_neb,
)

anchors = [12, 37]  # one atom per half, where the two are joined
base_a = bonded_cluster_indices_no_anchor_hub(atoms, anchors[0])
base_b = bonded_cluster_indices_no_anchor_hub(atoms, anchors[1])

product = get_best_flip_and_face_bases(atoms, base_a, base_b, anchors, calc=calc)
neb = prepare_neb(atoms, product, calc, n_images=7)
```

Pass `optimise_after=False` to skip the relaxation and get the rigid swap
alone, in which case no calculator is needed.

### Seeding a product from a transition state

The two functions above build a product out of what the reaction is known to
do. When that is not known, but a transition state is already to hand -- from a
scan, a database, or a saddle search that started from something else --
`seed_product_from_ts` builds one out of the saddle instead. It interpolates
from the reactant to the transition state geodesically, reads the direction the
path is travelling in as it arrives, and keeps stepping that way past it:

```python
from reactiontools import optimise_geom, prepare_neb, seed_product_from_ts

seed = seed_product_from_ts(reactant, ts)
product = optimise_geom(seed, calc)
neb = prepare_neb(reactant, product, calc, n_images=7)
```

Nothing is evaluated while the seed is built, so no calculator is involved
until the relaxation. `push` sets how far past the saddle to step, as a
multiple of the reactant-to-transition-state distance: the default of 1.0 lands
roughly where the reactant would be reflected through the saddle, which suits a
near-symmetric reaction such as a proton transfer, and a product further out
wants more.

Whether the seed landed in the intended basin is settled by relaxing it, not by
building it -- run the band above and check it comes back over a barrier near
the transition state it started from. `seed_product_from_ts` only warns
`SeedWarning` about what it can see for itself: a push stopped short to avoid
driving two atoms through each other, or one that went nowhere. Pass
`return_path=True` for the whole band it built, reactant through saddle to
seed, to plot or to hand to `restart_neb`.

`optimise_irc` answers the same question properly, by following the true
reaction coordinate downhill from a converged saddle, and costs hundreds of
gradients to do it.

### Running metadynamics

The three stages of a metadynamics run, in order: pick the atoms, build the
input, bias the dynamics with it, sum the hills.

```python
from ase import units
from ase.md.langevin import Langevin
from reactiontools import (
    find_molecules,
    plot_plumed,
    plumed_calculator,
    plumed_metad_input,
    plumed_selection,
    run_sum_hills,
)

solute, solvent = find_molecules(atoms)[:2]

lines = plumed_metad_input(
    cvs=[
        f"c1: COORDINATION GROUPA={plumed_selection(solute)} "
        f"GROUPB={plumed_selection(solvent)} R_0=3.0"
    ],
    sigma=0.05,
    height=0.02,
    pace=100,
    biasfactor=10,
    temperature=300,
)

with plumed_calculator(atoms, calc, lines, timestep=0.5 * units.fs, temperature=300):
    Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.01).run(100000)

run_sum_hills()
plot_plumed("fes.dat", x_label="Coordination")
```

`plumed_calculator` wraps your calculator so the integrator asks for forces as
usual and PLUMED adds the bias on top. It is a context manager because PLUMED
buffers what it writes and only flushes when finalised: run the dynamics inside
the block, or `HILLS` ends short of the hills actually deposited and the surface
summed from it is quietly wrong rather than obviously missing. The block puts
your original calculator back on the way out, exceptions included.

Three things to keep in step, none of which is checked for you:

- **The timestep** must be the one the integrator uses, since PLUMED counts its
  own steps from it.
- **The temperature** goes to `plumed_metad_input` as `TEMP`, to
  `plumed_calculator` as the thermal energy, and to the thermostat. All three
  should agree.
- **`biasfactor`** makes it well-tempered. Without one the Gaussians never stop
  piling up and there is no converged surface to read off.

`plumed_metad_input` writes `UNITS ENERGY=eV LENGTH=A TIME=fs` first, so
`sigma`, `height` and everything PLUMED writes back are in Å and eV rather than
PLUMED's own nm and kJ/mol. That is what makes `plot_plumed` right about the
file it reads. `PLUMED_ASE_UNITS` is the same line, for hand-written input.

`METAD` settings without their own argument go through `metad_extra` — the grid
ones matter for a long run — and further actions through `extra`:

```python
lines = plumed_metad_input(
    cvs=["d1: DISTANCE ATOMS=1,2"],
    sigma=0.05,
    height=0.02,
    pace=100,
    biasfactor=10,
    temperature=300,
    metad_extra="GRID_MIN=1.0 GRID_MAX=6.0 GRID_BIN=500",
    extra=["UPPER_WALLS ARG=d1 AT=6.0 KAPPA=100.0"],
)
```

CV lines go through untouched, so any PLUMED action works — and any mistake in
one is PLUMED's to report. It checks them as the calculator is built, before
the block is entered and so before anything is written, which is why a missing
`R_0` above comes back as `keyword R_0 is compulsory for this action` rather
than as a run that goes nowhere.

Only `plumed_calculator` needs the plumed Python module, and only
`run_sum_hills` needs the `plumed` executable. The rest is string handling.

### PLUMED post-processing

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

#### Reading the numbers off a surface

`summarise_fes` measures a 1-D profile: the barrier each way, and the free
energy of one basin relative to the other. Give it the two basins as windows on
the collective variable — reading them off the plot is the point:

```python
from reactiontools import summarise_fes

print(
    summarise_fes("fes.dat", basin_a=(1.0, 2.0), basin_b=(3.0, 4.5), source_unit="eV")
)
```

```
Barrier A->B:  0.352 eV
Barrier B->A:  0.418 eV
Delta F (B-A): -0.066 eV
Minima at:     1.5, 3.75
Barrier at:    2.6
```

Windows rather than automatic minimum-finding, because a noisy surface has
local minima everywhere, and because a convergence series only means anything
if every surface is measured the same way.

By default `delta_f` is the difference between the two minima. Pass a
`temperature` and it comes instead from integrating the Boltzmann weight across
each basin, which counts a wide basin as more probable than a narrow one of the
same depth:

```python
summarise_fes(
    "fes.dat", basin_a=(1.0, 2.0), basin_b=(3.0, 4.5), source_unit="eV", temperature=300
)
```

The barriers are measured out of the bottom of each well either way, since that
is what a barrier is.

#### Has it converged?

`stride` sums the hills every so often instead of once at the end, giving a
series of surfaces. The run has converged when the last few lie on top of each
other:

```python
from reactiontools import plot_fes_1d, run_sum_hills, sum_hills_files

run_sum_hills(stride=100, outfile="fes", grid_min=1.0, grid_max=6.0, grid_bin=500)

surfaces = sum_hills_files("fes")
plot_fes_1d(
    surfaces,
    source_unit="eV",
    max_datasets=5,
    labels=[(i + 1) * 100 for i in range(len(surfaces))],
    label_template="{:g} hills",
)
```

Set the grid explicitly for a series. Without it PLUMED picks bounds per
surface from the hills it has so far, and the surfaces come back on grids that
do not line up.

Surfaces lying on top of each other is the loose version of the test. The one
that matters is whether the numbers have stopped moving, which
`plot_fes_convergence` draws directly — the barrier and the basin difference
against time:

```python
from reactiontools import fes_convergence, plot_fes_convergence

hills = [(i + 1) * 100 for i in range(len(surfaces))]
plot_fes_convergence(
    surfaces,
    basin_a=(1.0, 2.0),
    basin_b=(3.0, 4.5),
    times=hills,
    source_unit="eV",
    filename="convergence",
)
```

`fes_convergence` returns the underlying `FESSummary` per surface if you want
the numbers rather than the picture.

`sum_hills_files` exists because the naming is a trap. `--stride` does not
number the file it was given — it writes `f"{outfile}{n}.dat"`, so the default
`outfile="fes.dat"` yields `fes.dat0.dat`, and the obvious glob `fes*.dat` is
right only by accident. Worse, sorting those names puts `fes.dat10.dat` before
`fes.dat2.dat`, which for a convergence series silently scrambles the very
thing being plotted. `sum_hills_files` orders them by the index PLUMED gave
them. Passing `outfile="fes"` at least gets you `fes0.dat`.

To project a surface onto some of its variables, name the ones to keep and give
the temperature the rest are integrated out at, in the energy units of the
hills — eV for a run built by `plumed_metad_input`:

```python
from ase.units import kB

run_sum_hills(idw="d1", kt=kB * 300, outfile="fes_d1.dat")
```

There has to be something left to integrate out, so this needs a run biased on
two variables or more. On a one-variable `HILLS` PLUMED does not report a
usable error — it aborts the process on a failed internal assertion,
`proj.size()<getNumberOfArguments()`, which surfaces here as a
`CalledProcessError` with a signal rather than an exit code.

Anything without its own keyword goes through `extra`, for example
`extra=["--fmt", "%14.9f"]`.

## API reference

### `tools_reaction` — building and analysing paths

| Function | Description |
| --- | --- |
| `get_neb_path(images)` | Cumulative reaction-path distance along a band, starting at zero. |
| `stitch_path(path1, path2, f_reverse_path=False)` | Join a reactant-side and product-side path into one IRC-like sequence. |
| `resample_path(path, n_resample)` | Cubic-spline resample a path to a fixed number of images, preserving the endpoints. |
| `optimise_geom(atoms, calc, ..., optimiser=BFGS, logfile='-', keep_traj=False)` | Relax a structure and return the final image. `use_socket=True` drives `calc` over an ASE `SocketIOCalculator`. |
| `optimise_reactant_product(reactant, product, calc, ..., use_socket=False)` | Relax both endpoints independently, one after the other. Reports the two separately. |
| `prepare_neb(reactant, product, calc, n_images=5, climb=True, geo_int=True, k=2.0, parallel=False)` | Build a configured `ase.mep.NEB`, interpolating geodesically or with IDPP. `parallel=True` evaluates images concurrently. |
| `restart_neb(images, calc, n_images=None, climb=True, ...)` | Build a band from one already relaxed, to continue it rather than interpolate afresh. `n_images` resamples on the way in. |
| `prepare_parallel_neb(reactant, product, make_calc, n_images=5, ...)` | Context manager giving each interior image its own socket calculator, so the images are evaluated concurrently. |
| `restart_parallel_neb(images, make_calc, n_images=None, ...)` | `restart_neb` over sockets, reusing the endpoint energies stored with the band. |
| `socket_calculators(n_calculators, make_calc=None, ...)` | Context manager opening a pool of `SocketIOCalculator`s, one socket each, closed on exit. |
| `optimise_neb(neb, fmax=0.01, steps=1000, ts_traj='ts.traj', optimiser=BFGS, logfile='-')` | Relax the band and return the final images. |
| `ConvergenceWarning`, `ConvergenceError` | Warned, or raised under `raise_on_unconverged=True`, when an `optimise_*` run hits its step limit. |
| `get_ts_image(neb_images, calc=None)` | The highest-energy image along a band, reusing the energies the images carry unless a calculator is given. |
| `summarise_neb(images, calc=None)` | Forward and reverse barriers, reaction energy and TS index, as a `NebSummary`. |
| `NebSummary` | What `summarise_neb` returns: `barrier`, `reverse_barrier`, `reaction_energy`, `ts_index`, `energies`, `is_barrierless`. Prints as a short report. |
| `get_fmax(atoms)` | Largest per-atom force, the quantity the optimisers converge against. |
| `optimise_ts(ts_image, calc, fmax=0.01, eta=1e-4, gamma=0.1, logfile='-', internal=False)` | Refine a TS guess to a true saddle point with Sella. |
| `optimise_irc(ts_image, calc, dx=0.1, ..., logfile='-')` | Follow the IRC downhill in both directions, returning `(forward, reverse)`. |
| `get_vibrations(atoms, calc)` | Finite-difference frequencies in cm⁻¹; one imaginary mode confirms a saddle. |
| `quick_guess_path(reactant, product, n_images=25)` | Geodesic path guess, no optimisation. |
| `quick_guess_ts(reactant, product, n_images=25)` | Midpoint of a geodesic guess, as a cheap TS starting structure. |

### `tools_orca` — ORCA calculators and conformer searches

Needs an ORCA install; see the dependency table above.

| Function | Description |
| --- | --- |
| `orca_calc_preset(calc_type='DFT', xc='r2SCAN-3c', charge=0, multiplicity=1, ...)` | Build an ASE ORCA calculator from presets for DFT, MP2, CCSD(T) or QM/XTB2, without hand-writing ORCA input. Drop the result into any function that takes a `calc`. |
| `orca_optimise_atoms(atoms, xc='r2SCAN-3c', tight_opt=True, ...)` | Relax a geometry with ORCA's own optimiser rather than ASE's, for molecules that BFGS in Cartesians struggles with. |
| `orca_calculate_goat(atoms, charge=0, multiplicity=1, n_procs=1, method='XTB')` | Run a GOAT conformer search, returning `(conformers, DataFrame)` of energies and populations. |

`f_solv` and `f_disp` take either `True` for the default (SMD water, D4) or a
string naming the solvent or dispersion keyword directly.

Worth running `orca_calculate_goat` before building a band: a NEB between two
arbitrary conformers explores the conformational change as well as the
reaction, and the barrier that comes back is not the one you wanted. It
searches at `method='XTB'` by default, because GOAT runs many optimisations
and the method has to be a cheap one.

#### Presets

Five dictionaries name levels of theory worth reaching for by habit. Splat one
into `orca_calc_preset`, and override anything on top of it — later keywords
win:

```python
from reactiontools import orca_calc_preset, orca_preset_dft_gold

calc = orca_calc_preset(**orca_preset_dft_gold, n_procs=8)
```

| Preset | Level of theory |
| --- | --- |
| `orca_preset_dft_cheap` | BLYP/6-31+G(d,p), gas phase. A first look, or the many single points of a scan. |
| `orca_preset_dft_gold` | B3LYP/def2-SVP with D4 dispersion, in implicit water. |
| `orca_preset_xtb` | GFN2-xTB. Fast enough to drive a NEB with, so the usual choice for a first band. |
| `orca_preset_mp2_gold` | DLPNO-MP2/def2-TZVPP in implicit water. |
| `orca_preset_ccsd_gold` | CCSD(T)/def2-TZVPP in implicit water, the reference to judge the others against. |

`orca_preset_ccsd_gold` sets `calc_type='CCSD(T)'`, which is passed straight
through to ORCA and so runs *canonical* CCSD(T). Pass `calc_type='CCSD'`
instead for the linear-scaling `DLPNO-CCSD(T)` approximation, which is the only
tractable option beyond a handful of atoms.

#### Tiered calculators

Beyond the presets sit three tiers of increasing cost, for taking a system
from a first screen to a publishable number: screen cheaply, refine the
mechanism at wB97M-V/def2-TZVPD, then put a CCSD(T)/CBS energy on the
stationary points. Every function in this module finds the binary through the
same resolver, which takes `orca_path=` explicitly or the `ORCA_PATH`
environment variable — naming either the executable or its install
directory — and refuses look-alikes such as the GNOME screen reader that
ships as `/usr/bin/orca` on many Linux systems.

| Function | Description |
| --- | --- |
| `orca_cheap_calculator(method='gfn2-xtb', charge=0, multiplicity=1, solvent=None, native='auto', ...)` | Screening calculator: GFN-FF/GFN1/GFN2-xTB — ORCA's native xTB by default, the external `xtb` binary (`$XTBEXE`) otherwise — and the Grimme "3c" composites. `CHEAP_METHODS` lists the aliases, `NATIVE_XTB_METHODS` the native keywords. Solvation defaults to ALPB for xTB and CPCM otherwise. |
| `orca_calculator(charge=0, multiplicity=1, task='engrad', ...)` | OMol25-level wB97M-V/def2-TZVPD calculator for mechanism work. `task` picks sp/engrad/opt/optts/neb-ts/irc/freq/scan and builds the matching `%geom`/`%neb`/`%irc`/`%freq` blocks; `'engrad'` is the only task ASE reads forces from. |
| `sella_ts_search(atoms, charge=0, multiplicity=1, fmax=0.02, internal=False, ...)` | Saddle search with Sella over `orca_calculator` gradients, avoiding the 6N-gradient numerical Hessian that ORCA's own OptTS would trigger. |
| `orca_gold_standard(atoms, directory='orca_gold', ...)` | Compound CCSD(T)/CBS focal-point job: an optional geometry + frequency stage, MP2/CBS extrapolation and a DLPNO-CCSD(T) delta, with finished stages reused across reruns. |
| `GoldStandard` | Its result dataclass: `e_hf_cbs`, `e_corr_cbs`, `e_total`, ZPE and thermal corrections, `energy`/`enthalpy`/`gibbs` properties and a `summary()` report. |
| `reaction_energy(reactants, products, thermo='gibbs')` | Difference two lists of `GoldStandard` results, in kcal/mol. |

### `tools_geometry` — building product end states

A NEB needs a product as well as a reactant, and the product is usually the
awkward one to draw by hand. These build it: superpose corresponding structures,
for a stacked dimer find the two halves and swap them over, or for a proton
transfer move the hydrogen across. When the mechanism is not known well enough
to draw the product but a transition state is already to hand,
`seed_product_from_ts` builds one from that instead.

| Function | Description |
| --- | --- |
| `kabsch_transform(mobile_positions, reference_positions, weights=None)` | Optimal proper rotation and translation between corresponding `(n, 3)` point sets. |
| `align_atom_sets(mobile, reference, mobile_indices=None, reference_indices=None, weights=None)` | Fit corresponding atoms and return a rigidly transformed copy of the whole mobile structure. Supports uniform, atomic-mass or custom weighting. |
| `atom_set_rmsd(mobile, reference, mobile_indices=None, reference_indices=None, weights=None, align=False)` | RMSD of corresponding selections in their current frames or after an optimal rigid fit. |
| `bonded_cluster_indices_no_anchor_hub(atoms, anchor, mult=1.0, multi_h=1.3)` | Atoms bonded to an anchor, without the walk routing back through it. |
| `get_dimer_bonded_cluster_indices(atoms, anchors, mults=None, multi_h=1.3)` | Union of the two clusters, one per anchor. |
| `flip_and_face_bases(atoms, baseA_idxs, baseB_idxs, anchors, rot_matrix=None)` | Swap two fragments over, each landing on the other's anchor and facing it. |
| `optimize_with_fixed_anchors(atoms, baseA_idxs, baseB_idxs, anchor_indices, calc, fmax=0.05, steps=1000, optimiser=BFGS, logfile='-')` | Relax the fragments with their anchors pinned, leaving all other atoms untouched. |
| `get_best_flip_and_face_bases(atoms, baseA_idxs, baseB_idxs, anchors, optimise_after=True, calc=None)` | Search the reflection signs and keep whichever leaves the fragment centres of mass closest. |
| `swap_bonding_configuration(atoms, donor_index, hydrogen_index, acceptor_index)` | Turn one or more donor-H...acceptor bonds into donor...H-acceptor, keeping each original donor-H length. Scalar donor/acceptor indices are shared; iterables pair one-to-one with the hydrogen indices. |
| `seed_product_from_ts(reactant, ts, n_images=25, push=1.0, n_steps=10, tangent_images=2, weights='masses', clash_scale=0.7, return_path=False)` | Seed the end state on the far side of a transition state: interpolate reactant to TS geodesically, take the direction the path arrives in, and keep stepping past the saddle. No calculator, and the seed comes back unrelaxed. Warns `SeedWarning` if the push had to stop short of a clash or went nowhere. |
| `SeedWarning` | Raised as a warning by `seed_product_from_ts`. Promote it with `warnings.simplefilter('error', SeedWarning)`. |

### `tools_plumed` — metadynamics support

| Function | Description |
| --- | --- |
| `plumed_selection(indices)` | Format zero-based atom indices as a one-based PLUMED selection with compact ranges. |
| `plumed_metad_input(cvs, sigma, height, pace, biasfactor=None, temperature=None, ...)` | Build the PLUMED input for a metadynamics run, in ASE's units. |
| `plumed_calculator(atoms, calc, input_lines, timestep, temperature=None, ...)` | Context manager wrapping `calc` in a PLUMED bias, so an ASE dynamics run becomes a biased one. Needs the plumed Python module. |
| `PLUMED_ASE_UNITS` | `"UNITS ENERGY=eV LENGTH=A TIME=fs"`, for hand-written input. |
| `find_molecules(atoms)` | Split a structure into connected components using an ASE neighbour list. |
| `run_sum_hills(hills='HILLS', outfile='fes.dat', mintozero=True, stride=None, grid_min=None, grid_max=None, grid_bin=None, idw=None, kt=None, ...)` | Run `plumed sum_hills` to build a free-energy surface, or a `stride`d series of them. |
| `sum_hills_files(outfile='fes.dat')` | The surfaces a strided run wrote, ordered by index rather than by name. |
| `run_opes_fes(state='STATE', outfile='fes.dat', grid_min=None, grid_max=None, grid_bin=None, kt=None, ...)` | The `OPES_METAD` counterpart: rebuild the surface from a `STATE` file with the bundled `FES_from_State.py`. |

`OPES_METAD` deposits no hills to add up, writing a running estimate of the
bias to a `STATE` file instead, so `run_opes_fes` is what reads a surface back
out of an OPES run. It runs the script under `sys.executable`, and through
`subprocess.run(check=True)` so a failed reconstruction is not mistaken for a
successful one.

### `tools_cv` — collective variables for proton transfer

Six builders that write the PLUMED input biasing a reaction, in two families: a
coordination difference for one or two proton transfers, and progress along a
reference path. Each returns the script and the shell command that turns the
resulting bias back into a free-energy surface. Below them sit the building
blocks for writing a collective variable of your own, which is where one that
encodes a particular system belongs.

| Function | Description |
| --- | --- |
| `plumed_input_1pt(geometry, idx, temperature, r_0=1.1, wall=1.5, angle_lim=130.0, ...)` | Bias one proton transfer. The CV is the donor/acceptor coordination difference, running +1 to −1 as the proton crosses. |
| `plumed_input_2pt_1d(geometry, idx1, idx2, temperature, ...)` | Two transfers averaged into one concerted coordinate. |
| `plumed_input_2pt_2d(geometry, idx1, idx2, temperature, ...)` | The same two kept as separate axes, so the surface resolves concerted from stepwise. |
| `plumed_input_neb_path(temperature, wall=0.1, lambda_val=250.0, neigh_size=8, ...)` | Bias `path.sss` along a `PATHMSD` reference, walling `path.zzz`. For reactions no single geometric coordinate describes. |
| `plumed_input_steered(cv_block, cv_start, cv_stop, steps, ...)` | Drag any CV with a `MOVINGRESTRAINT`, rather than biasing it. Returns the script and the total step count. |
| `plumed_input_steered_pt(geometry, idx, steps, ...)` | The same for a proton transfer, reading the pull's start and end off the geometry. |
| `switching_value(r, r_0, nn=6, mm=None)` | PLUMED's rational switching function in Python, for predicting what a coordination CV is worth without running it. |
| `as_positions(source)` | Coordinates in ångström from an `Atoms`, an OpenMM `Modeller`, an array or a file. |
| `plumed_bias_and_fes(f_opes, arg, pace, height, sigma, bias, temperature, kt, grid_bin, grid_min=None, grid_max=None, ...)` | The `METAD`/`OPES_METAD` line and the command that reads its surface back, built together so they cannot disagree. |
| `plumed_one_based(indices)` | Zero-based indices as PLUMED's one-based, order preserved. |
| `plumed_units_header(units)` | The `UNITS` line the script needs, or an empty string for PLUMED's own. |
| `plumed_temperature_pair(temperature, units)` | Kelvin for the bias `TEMP=`, and kBT in the script's energy unit for the reconstruction's `--kt`. |
| `plumed_angle_radians(degrees)` | A wall angle in the radians PLUMED wants. |

`geometry` is whatever holds one — an `ase.Atoms`, an `openmm.app.Modeller`, a
bare `(n, 3)` array or a path — and `temperature` is a number of kelvin or an
`openmm.unit.Quantity`. Every builder takes `f_opes=True` to swap
well-tempered `METAD` for `OPES_METAD`, which also swaps the returned command
from `plumed sum_hills` to the OPES one.

**Building your own.** A CV that encodes one particular system belongs with that
study rather than here, so the last five entries are public: write the CV lines
yourself and hand the label you biased to `plumed_bias_and_fes`, and what comes
out is a script indistinguishable from the ones above — including a
reconstruction command that agrees with the bias about the grid and the
temperature, which is the pairing that goes wrong when the two are written
separately.

```python
from reactiontools import (
    plumed_angle_radians,
    plumed_bias_and_fes,
    plumed_one_based,
    plumed_temperature_pair,
    plumed_units_header,
)

n1, h1, o2 = plumed_one_based([30, 31, 44])
kelvin, kt = plumed_temperature_pair(300.0, "plumed")
metad_line, fes_command = plumed_bias_and_fes(
    False,
    "z",
    pace=500,
    height=15.0,
    sigma=0.05,
    bias=20.0,
    temperature=kelvin,
    kt=kt,
    grid_bin=200,
    grid_min=-0.3,
    grid_max=0.3,
)

script = f"""{plumed_units_header("plumed")}n1_h1: DISTANCE ATOMS={n1},{h1}
o2_h1: DISTANCE ATOMS={o2},{h1}
z:     COMBINE ARG=n1_h1,o2_h1 COEFFICIENTS=1,-1 PERIODIC=NO
ang:   ANGLE ATOMS={n1},{h1},{o2}
       LOWER_WALLS ARG=ang AT={plumed_angle_radians(130.0)} KAPPA=500
{metad_line}
PRINT ARG=z,metad.bias STRIDE=500 FILE=COLVAR
"""
```

`plumed_one_based` is not `plumed_selection` from `tools_plumed`: that one sorts
and dedupes to collapse indices into ranges, which is right for the atom group
of a `COORDINATION` and wrong for anything positional like a donor, hydrogen and
acceptor read off in order.

**Units.** These scripts are for an external PLUMED, so `units="plumed"` (the
default) writes no `UNITS` line and works in nanometres and kJ/mol. Pass
`units="ase"` for ångström and eV with `PLUMED_ASE_UNITS` prepended. Note this
is the **opposite** default to `plumed_metad_input`, which only ever feeds
ASE's own `Plumed` calculator: a script moved between the two without changing
`units` is wrong by a factor of ten in every length. Lengths derived from the
geometry are converted for you; ones you supply are not.

### `tools_path` — reference paths from steered MD

Turning a steered-MD trajectory into the reference a `PATHMSD` collective
variable needs, which is the cheap alternative to relaxing a NEB and gives a
path in the full solvated environment rather than one interpolated in vacuum.

| Function | Description |
| --- | --- |
| `path_from_steered_md(traj_file, template_pdb='index_atoms.pdb', output_file='neb_path.pdb', colvar_file='COLVAR_SMD', n_images=15, smooth=0, ...)` | Pick frames evenly spaced along the CV, align and smooth them, write the multi-model PDB, and return the recommended `LAMBDA`. |
| `estimate_path_lambda(pdb_path, length_unit='nm')` | Size the `LAMBDA` a path should be given, from the mean squared displacement between its frames. |
| `select_frames_by_cv(cv, n_images, cv_start=None, cv_stop=None)` | Frame indices evenly spaced along a collective variable, never going backwards. |
| `select_frames_by_msd(xyz, n_images)` | Frame indices evenly spaced along the trajectory's own arc length, for when there is no COLVAR. |
| `cv_from_colvar(colvar_file, n_frames, cv_name=None)` | One CV value per trajectory frame, dropping the row PLUMED writes at step 0. |

`LAMBDA` has units of inverse squared length, so `length_unit` matters: the
answer for an ångström-based run is a hundred times smaller than for a
nanometre-based one.

Trajectory readers use MDTraj, which is installed with `reactiontools` and
imported only when needed. The three selection functions remain NumPy-only.

### `tools_io` — structure files

| Function | Description |
| --- | --- |
| `convert_xyz_to_plumed_ref(xyz_file, template_pdb, output_file, atom_line='HETATM')` | Splice a path's coordinates into a template's atom records to make the multi-model PDB `PATHMSD` reads. Renumbers both so their serials agree, which PLUMED insists on. |
| `pdb_remove_ter_index(input_path, output_path)` | Renumber atom serials from 1 per model, keeping `TER` and `CONECT` in step. |
| `strip_hydrogens_keep_indices(input_pdb, output_pdb, keep=None)` | Drop every hydrogen but the named ones, so a path CV measures the reaction rather than the thermal noise. |
| `convert_xyz_to_pdb(input_file, output_file, cutoff_multiplier=1.1, index=-1)` | Perceive bonds by distance, assign chains and residues, write `CONECT` records. |
| `convert_pdb_to_xyz(input_file, output_file, comment=None)` | One XYZ frame per PDB model. |
| `element_from_pdb_line(line)` | The element of an `ATOM`/`HETATM` record, from the element column or the atom name's alignment. |
| `format_pdb_atom_name(symbol, count)` | A unique four-character atom name, indented as the PDB convention requires. |
| `write_xyz_frame(fh, symbols, positions, comment='')` | Write one XYZ frame to an open handle. |

`atom_line` is `'HETATM'` by default, which is what OpenMM writes for
ligand-like residues; ASE writes `'ATOM'`.

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
| `fes_series_files(directory='.', pattern=r'^fes_?(\d+)\.dat$')` | The numbered surfaces in a directory, ordered by index rather than by name — `fes_10` sorts before `fes_2`. |
| `load_fes_series(directory='.', energy_unit='eV', source_unit='kJ/mol', ...)` | The same, loaded through `as_fes`, ready for `plot_fes_1d` or `fes_convergence`. |
| `summarise_fes(source, basin_a, basin_b, temperature=None, ...)` | Barriers each way and the basin free-energy difference, as a `FESSummary`. |
| `FESSummary` | What `summarise_fes` returns: `forward_barrier`, `reverse_barrier`, `delta_f`, `minimum_a`/`minimum_b`, `depth_a`/`depth_b`, `barrier_position`. Prints as a short report. |
| `fes_convergence(sources, basin_a, basin_b, ...)` | One `FESSummary` per surface of a series. |
| `plot_fes_convergence(sources, basin_a, basin_b, times=None, ...)` | The barrier and the basin difference against time. |
| `plot_fes(sources, **kwargs)` | Plot, dispatching on dimensionality. |
| `plot_fes_1d(sources, labels=None, energy_unit=None, ...)` | One or many 1-D profiles on one axes. |
| `plot_fes_2d(sources, levels=30, cmap=None, ...)` | Filled contours, one panel per surface. |
| `plot_fes_path(source, path, path_columns=None, ...)` | A trajectory or reaction path through CV space over a filled 2-D surface. |
| `plot_fes_2d_overlay(sources, ...)` | Several 2-D surfaces as contour lines on shared axes. |
| `plot_fes_slices(sources, ...)` | 1-D cuts through a 2-D surface. |
| `plot_plumed_fes(path, ...)` | Convenience wrapper over `plot_fes` for a single file. |
| `plot_plumed_colvar(path, x_axis='time', columns=None, ...)` | One stacked panel per collective variable in a `COLVAR`. |

Energies are read as kJ/mol unless `source_unit` says otherwise, because that
is what PLUMED writes when driven from OpenMM. `max_energy` masks poorly
sampled regions rather than letting them dominate the colour scale, and
`filename=None` means write nothing.

Overlay the CV history from a PLUMED trajectory on its reconstructed surface:

```python
from reactiontools import plot_fes_path

plot_fes_path(
    "fes.dat",
    "COLVAR",
    path_label="MD trajectory",
    filename="fes-path",
)
```

When the `COLVAR` fields match the FES axes they are selected automatically;
otherwise pass `path_columns=("cv1", "cv2")`. An `(n_points, 2)` array can be
used in place of the file for a NEB or other precomputed path.

### `tools_units` — energy units and kBT

| Function | Description |
| --- | --- |
| `convert_energy(values, source, target)` | Convert between the units in `ENERGY_UNITS` (kJ/mol, kcal/mol, eV, meV, hartree, kT300). |
| `thermal_energy(temperature, energy_unit='kJ/mol')` | kBT at a temperature — the `--kt` that `sum_hills` and the OPES scripts reweight with. |
| `as_kelvin(temperature)` | A temperature in kelvin, from a number or an `openmm.unit.Quantity`. |
| `unit_label(unit)` | The LaTeX axis label for an energy unit. |
| `ENERGY_UNITS`, `DEFAULT_ENERGY_UNIT` | The units known, sized in kJ/mol, and the one assumed when nothing is said. |

These live apart from `tools_fes`, which re-exports them, so that the script
builders can convert energies without importing matplotlib.

### `tools_plotting` — figures

| Function | Description |
| --- | --- |
| `n_plot(xlab, ylab)` | Apply the house style to the current pyplot axes. |
| `ax_plot(fig, ax, xlab, ylab)` | Same, for an explicit `Figure`/`Axes` pair. `None` for either label leaves it untouched. |
| `plot_images(images, view='tilted', n_cols=4, ...)` | Grid of rendered structures, one panel per image. |
| `show_atoms(atoms, view='tilted', ...)` | Structures superimposed on one axes, for seeing how far a band has moved. |
| `plot_neb(images, calc=None, smooth=True, annotate=False, ...)` | NEB energy profile in meV against path length. `annotate=True` writes the barrier on the axes. |
| `plot_irc(images, calc=None, color='black', ...)` | The same profile with IRC defaults; pair with `stitch_path`. |
| `plot_temperature(trajectories, labels=None, timestep=None, ax=None)` | Temperature against frame or time for one or more trajectories. |
| `plot_total_energy(trajectories, labels=None, timestep=None, ax=None)` | Total energy against frame or time. |
| `plot_plumed(file='fes.dat', ...)` | One-dimensional PLUMED free-energy surface. |
| `plot_plumed_multi(files, mintozero=False, ...)` | Several surfaces overlaid; directories expand to the `fes.dat` files beneath them. |

`plot_plumed` and `plot_plumed_multi` are ASE-flavoured wrappers over
`tools_fes`: they assume `fes.dat` is in eV and plot meV. That holds when the
run declared `UNITS ENERGY=eV`, as `plumed_metad_input` does by default. It is
**not** what PLUMED does otherwise: left to itself it reads and writes its own
units, kJ/mol and nm, whatever drove it. For a file in kJ/mol — from OpenMM, or
from an ASE run whose input had no `UNITS` line — use `tools_fes` directly and
set `source_unit`.

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

`tools_fes` is the exception, because PLUMED's units depend on what the input
asked for rather than on what drove it: it defaults to kJ/mol, and any of the
units in `ENERGY_UNITS` can be selected per call with `source_unit` and
`energy_unit`. `plumed_metad_input` puts `UNITS ENERGY=eV LENGTH=A TIME=fs` at
the top of the input it builds, which is what keeps a run driven from here in
the same eV and Å as everything else.

`tools_cv` defaults the other way, to PLUMED's own nm and kJ/mol, because the
scripts it writes are for an external PLUMED driven from OpenMM rather than for
ASE's `Plumed` calculator. **The two defaults are opposite, and nothing checks
that a script matches the run it is given to**: taking one built by
`plumed_input_1pt` into an ASE run, or the reverse, is wrong by a factor of ten
in every length and by 96.5 in every energy. Pass `units="ase"` to `tools_cv`
for the ASE convention, and remember that lengths you supply yourself — `wall`
in `plumed_input_neb_path`, and anything you interpolate into a CV block of your
own — are in whichever you chose. Lengths taken off the geometry are converted
for you; these are not, and nothing checks them.

## Testing

```bash
pytest --cov
```

The suite builds its own structures with `ase.build` and evaluates them with
EMT. Offline unit tests cover orchestration without opening sockets; a small
set of `integration` tests exercises real local socket transport when the
runner permits it. ORCA and OpenMM checks skip when those optional dependencies
are unavailable. Coverage is branch-aware and enforces the configured floor.

## Citing

If `reactiontools` is useful in your work, please cite it and whichever of
the codes it wraps you actually exercised — all in
[CITATIONS.bib](CITATIONS.bib):

| Entry | Cite for | Used by |
| --- | --- | --- |
| `Slocombe_reactiontools` | `reactiontools` itself | always |
| `larsen2017atomic` | [ASE](https://wiki.fysik.dtu.dk/ase/) | NEB, optimisation and I/O throughout |
| `zhu2019geodesic` | [`geodesic_interpolate`](https://github.com/LouieSlocombe/geodesic_interpolate) | `prepare_neb(geo_int=True)`, `quick_guess_path`, `quick_guess_ts`, `seed_product_from_ts` |
| `hermes2022sella` | [Sella](https://github.com/zadorlab/sella) | `optimise_ts`, `optimise_irc`, `sella_ts_search` |
| `plumed2` | [PLUMED](https://www.plumed.org/) | `run_sum_hills`, `plumed_calculator` |
| `laio2002escaping` | The metadynamics method | `plumed_metad_input` |
| `barducci2008well` | Well-tempered metadynamics | `plumed_metad_input(biasfactor=...)` |
| `jonsson1998nudged`, `henkelman2000improved`, `henkelman2000climbing` | The NEB method, the improved tangent and the climbing image | `prepare_neb`, `optimise_neb` |
| `smidstrup2014improved` | IDPP interpolation | `prepare_neb(geo_int=False)` |
| `nocedal2006numerical` | BFGS | every `optimise_*` that is not Sella |
| `neese2012orca`, `neese2022orca5`, `neese2025orca6` | [ORCA](https://www.faccts.de/orca/) | everything in `tools_orca` |
| `desouza2025goat` | The GOAT conformer search | `orca_calculate_goat` |
| `grimme2021r2scan3c`, `furness2020r2scan` | The default `r2SCAN-3c` functional | `orca_calc_preset`, `orca_optimise_atoms`, `orca_gold_standard(opt_method=...)` |
| `caldeweyher2019d4` | D4 dispersion | `f_disp=True` |
| `barone1998cpcm`, `marenich2009smd` | CPCM/SMD implicit solvation | `f_solv`, `solvent=` |
| `riplinger2013efficient`, `riplinger2013natural`, `pinski2015sparse` | DLPNO-MP2 and DLPNO-CCSD(T) | `calc_type='MP2'`, `calc_type='CCSD'`, `orca_gold_standard` |
| `bannwarth2019gfn2` | GFN2-xTB | `calc_type='QM/XTB2'`, `orca_cheap_calculator` |
| `spicher2020gfnff` | GFN-FF | `orca_cheap_calculator(method='gfn-ff')` |
| `ehlert2021alpb` | ALPB implicit solvation | `orca_cheap_calculator(solvent=...)` at the xTB levels |
| `mardirossian2016wb97mv` | The wB97M-V functional | `orca_calculator`, `sella_ts_search` |

## License

MIT — see [LICENSE](LICENSE).
