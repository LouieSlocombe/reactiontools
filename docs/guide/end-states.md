# Building end states

A band needs two endpoints. These are the two ways of getting the second
one when you have only the first: build it from what the reaction is known
to do, or step past a transition state that is already to hand.

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

`optimise_irc` answers the same question properly, by following the true
reaction coordinate downhill from a converged saddle, and costs hundreds of
gradients to do it.
