# Units

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
