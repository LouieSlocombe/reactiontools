# Continuing a band

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
