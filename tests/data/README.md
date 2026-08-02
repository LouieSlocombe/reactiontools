# Test data

Reference files for the test suite.

The tests are deliberately self-contained: structures are built on the fly with
`ase.build` and evaluated with the EMT calculator, and trajectories are written
to `tmp_path`. Nothing here is required for `pytest` to pass.

Use this directory only for fixtures that genuinely cannot be generated, such as
a recorded PLUMED `HILLS` file or a `fes.dat` from a production run. Keep them
small — a few kilobytes at most — since they are committed to the repository.
