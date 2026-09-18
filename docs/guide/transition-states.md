# Refining the transition state

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

These two use Sella, which is part of the package as `tools_sella`. Nothing
extra has to be installed to run them.

`seed_minima_from_ts` answers the same question as `optimise_irc` without
either: it rattles the saddle at random and relaxes what comes out, from enough
directions to land in both of the basins either side of it. That costs a
handful of geometry relaxations rather than hundreds of gradients, and does not
need a tightly converged saddle -- but it follows no reaction coordinate, so
what it finds is where the structure rolled, not what the saddle connects.
See [Building end states](end-states.md).
