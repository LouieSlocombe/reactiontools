# Building end states

A band needs two endpoints. These are the ways of getting the second one when
you have only the first: build it from what the reaction is known to do, or
step past a transition state that is already to hand. The last of them needs no
end state at all -- given a saddle and a calculator, it finds both.

## Building a flipped end state

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

## Seeding a product from a transition state

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

## Rattling a transition state into both end states

`seed_product_from_ts` still needs an end state to read a direction off.
`seed_minima_from_ts` needs only the saddle. It displaces the structure at
random, relaxes what comes out, and repeats: a saddle is downhill in one
direction and uphill in every other, so a structure nudged off it rolls into
one of the two states it connects.

Each direction is stepped both ways, because a displacement and its negative
roll to *opposite* sides of a saddle. That is what brackets the reaction rather
than sampling one basin twice, and the pair the search most often brackets is
what comes back:

```python
from reactiontools import prepare_neb, seed_minima_from_ts

summary = seed_minima_from_ts(ts, calc)
print(summary)

reactant, product = summary.connecting_minima
neb = prepare_neb(reactant, product, calc, n_images=7)
```

```
TS energy:     3.689 eV
TS max force:  0.005 eV/A
Minima found:  2
  0  barrier 0.374 eV   0.403 A from the TS   3 of 6
  1  barrier 0.372 eV   0.398 A from the TS   3 of 6
Connecting:    minima 0 and 1
Discarded:     0 stalled, 0 unconverged, 0 failed
```

`stdev` is the one knob that matters. Too small and every relaxation converges
straight back onto the saddle, because the force there is already below any
`fmax` worth asking for -- so loosening `fmax` cannot rescue it, and the
`SeedWarning` you get says to raise `stdev` instead. ASE's own `Atoms.rattle`
defaults to 0.001 Å, which would stall every time; the 0.1 Å default here
clears a converged saddle reliably. Too large and the structure lands somewhere
the saddle never connected to.

The cost is `2 * n_directions` relaxations, ten by default. `indices` is worth
setting for anything large, on two counts: an isotropic rattle otherwise spends
most of its amplitude on atoms with nothing to do with the reaction, and the
RMSD that tells two minima apart dilutes as one over the square root of the
atom count until a real hop measures smaller than the scatter a loose `fmax`
leaves behind.

`align` decides whether rigid drift is fitted out before any of that is
measured, and is inferred: off for a periodic cell or a pinned substrate, whose
frame is already fixed, and on for a free molecule, which would otherwise have
one basin fragment into several as the rattle's leftover translation and
rotation -- which no optimiser removes, the force along a rigid mode being zero
-- pile up.

Nothing here checks that `ts` is a saddle; `get_vibrations` does that, and costs
more than the whole search. `ts_fmax` on the summary is the cheap stand-in.

`optimise_irc` answers the same question properly, by following the true
reaction coordinate downhill from a converged saddle, and costs hundreds of
gradients to do it.
