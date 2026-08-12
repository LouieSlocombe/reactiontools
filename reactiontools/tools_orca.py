"""ORCA calculators and post-processing.

ORCA is licensed separately and is not installable from PyPI or conda, so it
has to be put on the machine by hand; the functions here find it through the
``ORCA_PATH`` environment variable unless a path is passed explicitly. See
``build_tools/README.md`` for the install steps.

:func:`orca_calc_preset` is the entry point: it builds an ASE calculator from a
handful of presets, so a DFT, MP2, CCSD(T) or QM/XTB2 job can be set up without
hand-writing ORCA's input syntax. That calculator drops straight into the
reaction-path functions in :mod:`reactiontools.tools_reaction`. The other two
functions run ORCA's own drivers instead of ASE's: :func:`orca_optimise_atoms`
for a geometry optimisation and :func:`orca_calculate_goat` for a GOAT
conformer search.
"""

import os
import re
import tempfile
from pathlib import Path

import pandas as pd
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read


def orca_calc_preset(orca_path=None,
                     directory=None,
                     calc_type='DFT',
                     xc='r2SCAN-3c',
                     charge=0,
                     multiplicity=1,
                     basis_set='',
                     n_procs=1,
                     f_solv=False,
                     f_disp=False,
                     atom_list=None,
                     calc_extra=None,
                     blocks_extra=None,
                     scf_option=None):
    """Build an ASE ORCA calculator from a small set of common presets.

    Assembles the ORCA "simple input" line and block section for one of a
    few common calculation types, so callers do not have to hand-write ORCA
    input syntax for routine DFT, MP2, CCSD(T) or QM/XTB2 jobs.

    Parameters
    ----------
    orca_path : str or None, optional
        Path to the ORCA executable. If None, read from the ``ORCA_PATH``
        environment variable.
    directory : str or None, optional
        Working directory for ORCA's input/output files. If None, a new
        temporary directory is created.
    calc_type : str, optional
        One of ``'DFT'``, ``'MP2'``, ``'CCSD'`` or ``'QM/XTB2'``, each
        building the corresponding ORCA method keyword(s). Any other value
        is passed straight through as the ORCA method keyword.
    xc : str, optional
        Exchange-correlation functional, used for ``'DFT'`` and as the QM
        region's method for ``'QM/XTB2'``.
    charge : int, optional
        Total charge.
    multiplicity : int, optional
        Spin multiplicity. Values above 1 switch ``'DFT'``/``'QM/XTB2'`` to
        a ``UKS`` reference and ``'MP2'``/``'CCSD'`` to a ``UKS`` reference
        as well.
    basis_set : str, optional
        Basis set keyword, appended to the simple input line (and, for
        ``'MP2'``/``'CCSD'``, also used as the auxiliary ``/C`` basis).
    n_procs : int, optional
        Number of MPI processes requested via ``%pal``.
    f_solv : bool or str, optional
        Implicit solvation via CPCM/SMD. ``True`` uses water; a string names
        an explicit SMD solvent; ``False``/``None`` disables solvation.
    f_disp : bool or str, optional
        Dispersion correction. ``True`` uses ``D4``; a string is used as the
        dispersion keyword directly; ``False``/``None`` disables it.
    atom_list : str or None, optional
        ORCA atom-selection string for the QM region (without braces), used
        only when *calc_type* is ``'QM/XTB2'``.
    calc_extra : str or None, optional
        Extra text appended to the simple input line, e.g. ``'TIGHTOPT'``.
    blocks_extra : str or None, optional
        Extra ORCA block text appended after ``%pal``/``%CPCM``. Ignored
        when *calc_type* is ``'QM/XTB2'``, which builds its own blocks
        instead.
    scf_option : str or None, optional
        Extra SCF-related keyword appended to the simple input line.

    Returns
    -------
    ase.calculators.orca.ORCA
        A configured ORCA calculator requesting an energy and gradient
        (``EnGrad``).
    """
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    if directory is None:
        directory = os.path.join(tempfile.mkdtemp(), 'orca')

    profile = OrcaProfile(command=orca_path)
    inpt_procs = f'%pal nprocs {n_procs} end' if n_procs > 1 else ''

    if f_solv is not None and f_solv is not False:
        # `is True`, not truthiness: a solvent name is truthy too, and testing
        # it that way overwrote every name with WATER.
        if f_solv is True:
            f_solv = 'WATER'
        inpt_solv = (f'\n%CPCM SMD TRUE\n'
                     f'    SMDSOLVENT "{f_solv}"\n'
                     f'END')
    else:
        inpt_solv = ''

    if f_disp is None or f_disp is False:
        inpt_disp = ''
    else:
        inpt_disp = 'D4' if f_disp is True else f_disp

    if atom_list is not None and calc_type == 'QM/XTB2':
        inpt_xtb = f'\n%QMMM QMATOMS {{{atom_list}}} END END\n'
    else:
        inpt_xtb = ''

    if blocks_extra is None:
        blocks_extra = ''

    inpt_blocks = inpt_procs + inpt_solv + blocks_extra

    if calc_type == 'DFT':
        inpt_simple = f'{xc} {inpt_disp} {basis_set}'
    elif calc_type == 'MP2':
        inpt_simple = f'DLPNO-{calc_type} {basis_set} {basis_set}/C'
    elif calc_type == 'CCSD':
        inpt_simple = f'DLPNO-{calc_type}(T) {basis_set} {basis_set}/C'
    elif calc_type == 'QM/XTB2':
        inpt_simple = f'{calc_type} {xc} {inpt_disp} {basis_set}'
        inpt_blocks = inpt_procs + inpt_solv + inpt_xtb
    else:
        inpt_simple = f'{calc_type} {basis_set}'

    # NOTE: open-shell MP2/CCSD conventionally use a UHF reference;
    # UKS is kept for all methods to preserve existing behaviour.
    if multiplicity > 1 and calc_type in ('DFT', 'QM/XTB2', 'MP2', 'CCSD'):
        inpt_simple = 'UKS ' + inpt_simple

    if scf_option is not None:
        inpt_simple += ' ' + scf_option

    if calc_extra is not None:
        inpt_simple += ' ' + calc_extra

    return ORCA(
        profile=profile,
        charge=charge,
        mult=multiplicity,
        directory=directory,
        orcasimpleinput=inpt_simple + ' EnGrad',
        orcablocks=inpt_blocks
    )


def orca_optimise_atoms(atoms,
                        charge=0,
                        multiplicity=1,
                        orca_path=None,
                        xc='r2SCAN-3c',
                        basis_set='',
                        tight_opt=True,
                        tight_scf=False,
                        f_solv=False,
                        f_disp=False,
                        n_procs=1):
    """Optimise a geometry at the DFT level with ORCA.

    Builds a DFT calculator via :func:`orca_calc_preset` with an
    ``OPT``/``TIGHTOPT`` keyword, runs it in a scratch directory, and reads
    back the final geometry from ORCA's own optimisation trajectory.

    Unlike :func:`~reactiontools.tools_reaction.optimise_geom`, which drives
    an ASE optimiser over a calculator, this hands the whole relaxation to
    ORCA. Use it when ORCA's internal coordinates converge a molecule that
    BFGS in Cartesians struggles with.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry. Its ``calc`` is set to the ORCA calculator as a
        side effect.
    charge : int, optional
        Total charge.
    multiplicity : int, optional
        Spin multiplicity.
    orca_path : str or None, optional
        Path to the ORCA executable. If None, read from the ``ORCA_PATH``
        environment variable.
    xc : str, optional
        Exchange-correlation functional.
    basis_set : str, optional
        Basis set keyword.
    tight_opt : bool, optional
        If True, use ``TIGHTOPT`` instead of ``OPT``.
    tight_scf : bool, optional
        If True, add ``TIGHTSCF`` to the optimisation keywords.
    f_solv : bool or str, optional
        Implicit solvation; see :func:`orca_calc_preset`.
    f_disp : bool or str, optional
        Dispersion correction; see :func:`orca_calc_preset`.
    n_procs : int, optional
        Number of MPI processes requested via ``%pal``.

    Returns
    -------
    ase.Atoms
        The optimised geometry, read from ORCA's ``orca.xyz`` output.
    """
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    else:
        orca_path = os.path.abspath(orca_path)

    opt_option = 'TIGHTOPT' if tight_opt else 'OPT'
    calc_extra = f'{opt_option} TIGHTSCF' if tight_scf else opt_option

    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, "orca.xyz")
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra)
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        return read(orca_file, format="xyz")


def _extract_conformer_info(filepath: str | Path):
    """Parse the conformer ensemble table from an ORCA GOAT output file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the ORCA ``.out`` file from a GOAT run.

    Returns
    -------
    pandas.DataFrame
        One row per conformer, with columns ``Conformer`` (integer index),
        ``Energy_kcal_mol`` and ``Percent_total``.

    Raises
    ------
    ValueError
        If no ensemble table could be found in the file.
    """
    line_pat = re.compile(
        r"""^\s*
            (?P<conformer>\d+)\s+          # integer index
            (?P<energy>-?\d+\.\d+)\s+      # energy in kcal/mol
            \d+\s+                         # degeneracy (ignored)
            (?P<ptotal>\d+\.\d+)\s+        # % total
            \d+\.\d+\s*?$                  # % cumulative (ignored)
        """,
        re.VERBOSE,
    )
    header_pat = re.compile(r"Conformer\s+Energy.*% total", re.I)
    rows = []
    in_table = False
    with open(filepath, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not in_table and header_pat.search(line):
                in_table = True
                continue

            if in_table:
                if line.strip() == "" or line.strip().startswith("Conformers"):
                    break
                m = line_pat.match(line)
                if m:
                    rows.append(
                        (
                            int(m["conformer"]),
                            float(m["energy"]),
                            float(m["ptotal"]),
                        )
                    )
    if not rows:
        raise ValueError(
            "Could not locate ensemble table. Check that the file is complete."
        )

    return pd.DataFrame(
        rows, columns=["Conformer", "Energy_kcal_mol", "Percent_total"]
    )


def orca_calculate_goat(atoms,
                        charge=0,
                        multiplicity=1,
                        orca_path=None,
                        n_procs=1):
    """Run ORCA's GOAT conformer search and collect the resulting ensemble.

    Worth running before a band is built: a NEB between two arbitrary
    conformers explores the conformational change as well as the reaction,
    and the barrier that comes back is not the one you wanted.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry for the conformer search.
    charge : int, optional
        Total charge.
    multiplicity : int, optional
        Spin multiplicity.
    orca_path : str or None, optional
        Path to the ORCA executable. If None, read from the ``ORCA_PATH``
        environment variable.
    n_procs : int, optional
        Number of MPI processes requested via ``%pal``.

    Returns
    -------
    tuple
        ``(atoms, df)``: every conformer in the final ensemble, in the order
        ORCA wrote them, and their energies and populations as parsed by
        :func:`_extract_conformer_info`.
    """
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    else:
        orca_path = os.path.abspath(orca_path)
    profile = OrcaProfile(command=orca_path)
    inpt_procs = f'%pal nprocs {n_procs} end' if n_procs > 1 else ''

    with tempfile.TemporaryDirectory() as temp_dir:
        calc = ORCA(
            profile=profile,
            charge=charge,
            mult=multiplicity,
            directory=temp_dir,
            orcasimpleinput='GOAT',
            orcablocks=inpt_procs
        )
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        xyz_file = os.path.join(temp_dir, "orca.finalensemble.xyz")
        orca_file = os.path.join(temp_dir, "orca.out")

        df = _extract_conformer_info(orca_file)
        atoms = read(xyz_file, format="xyz", index=':')
        return atoms, df
