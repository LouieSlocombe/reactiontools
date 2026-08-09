# Test data

Reference files for the test suite.

The tests are deliberately self-contained: structures are built on the fly with
`ase.build` and evaluated with the EMT calculator, and trajectories are written
to `tmp_path`. Nothing here is required for `pytest` to pass.

Use this directory only for fixtures that genuinely cannot be generated, such as
a recorded PLUMED `HILLS` file or a `fes.dat` from a production run. Keep them
small — a few kilobytes at most — since they are committed to the repository.

| File | Used by | Why it is committed |
| --- | --- | --- |
| `fad.xyz` | `test_orca.py` | Formic acid dimer. The ORCA tests assert absolute energies at `r2SCAN-3c`, so they are tied to this exact geometry — regenerating it would invalidate the reference values. Those tests skip unless `ORCA_PATH` is set. |
