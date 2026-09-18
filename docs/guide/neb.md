# Nudged elastic bands

Everything between building a band and trusting the number that comes off
it: which settings `prepare_neb` should be given, how to read a relaxed
band, how to tell whether it really converged, and which optimiser to relax
it with.

## Choosing NEB settings

`prepare_neb`'s defaults suit isolated molecules. A few are worth thinking
about:

- **`rm_ro_trans`** removes rigid-body rotation and translation. That is right
  for a molecule tumbling in vacuum, but wrong for a periodic slab whose atoms
  are already pinned by a constraint — leaving it on stops the band converging.
  Set `rm_ro_trans=False` for constrained or periodic systems.
- **`geo_int`** uses geodesic interpolation, which finds the shortest path in
  a metric of scaled inter-atomic distances rather than in Cartesian space, so
  atoms do not run through each other on the way across. With `geo_int=False`
  the band is built by linear interpolation refined with IDPP, which is usually
  fine for a small displacement like the adatom hop in the
  [quickstart](../quickstart.md). `prepare_neb` takes the interpolation as it
  comes; call
  [`geodesic_interpolate`](../api/tools_geodesic.md) yourself and hand the
  images to `optimise_neb` if you need to tune the metric, the tolerance or
  the seed.
- **`parallel`** evaluates the interior images concurrently instead of one at
  a time, for both the initial energies and every force call `optimise_neb`
  makes afterwards. Without an MPI launcher this runs each image's calculator
  in its own thread, which only helps if `calc` releases the GIL while it
  runs (e.g. one that shells out to an external code — a Python-only
  calculator like EMT gains nothing). Run under `mpirun` and ASE instead
  distributes the images across MPI ranks; pass a specific communicator with
  `world` if you don't want `ase.parallel.world`.

## Reading the numbers off a band

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

## Knowing whether it converged

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

## Choosing the optimiser, and where it logs

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
