# Citing

If `reactiontools` is useful in your work, please cite it and whichever of
the codes it wraps you actually exercised — all in
[CITATIONS.bib](https://github.com/LouieSlocombe/reactiontools/blob/main/CITATIONS.bib):

| Entry | Cite for | Used by |
| --- | --- | --- |
| `Slocombe_reactiontools` | `reactiontools` itself | always |
| `larsen2017atomic` | [ASE](https://wiki.fysik.dtu.dk/ase/) | NEB, optimisation and I/O throughout |
| `zhu2019geodesic` | [`geodesic-interpolate`](https://github.com/virtualzx-nad/geodesic-interpolate), which `tools_geodesic` is derived from | `geodesic_interpolate`, `prepare_neb(geo_int=True)`, `quick_guess_path`, `quick_guess_ts`, `seed_product_from_ts` |
| `hermes2022sella` | [Sella](https://github.com/zadorlab/sella), which `tools_sella` is derived from | `optimise_ts`, `optimise_irc`, `sella_ts_search`, `Sella`, `IRC` |
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
