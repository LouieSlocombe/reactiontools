"""ORCA calculators for ASE, from presets and cheap screening up to CCSD(T)/CBS.

ORCA is licensed separately and is not installable from PyPI or conda, so it
has to be put on the machine by hand; everything here locates the binary
through :func:`_resolve_orca`, which honours an explicit ``orca_path=``
argument, then ``$ASE_ORCA_COMMAND``, ``$ORCA_COMMAND``, ``$ORCA_PATH`` (the
executable itself or its install directory), ``$ORCA_DIR`` and finally PATH --
refusing anything that is not really ORCA, such as the GNOME screen reader
that ships as ``/usr/bin/orca`` on many Linux systems. See
``build_tools/README.md`` for the install steps.

:func:`orca_calc_preset` builds an ASE calculator from a handful of presets,
so a DFT, MP2, CCSD(T) or QM/XTB2 job can be set up without hand-writing
ORCA's input syntax. That calculator drops straight into the reaction-path
functions in :mod:`reactiontools.tools_reaction`. The ``orca_preset_*``
dictionaries below name a few levels of theory worth reaching for by habit,
and are splatted into it. :func:`orca_optimise_atoms` and
:func:`orca_calculate_goat` run ORCA's own drivers instead of ASE's, for a
geometry optimisation and a GOAT conformer search respectively.

Beyond the presets sit three tiers of increasing cost, cheapest first:

``orca_cheap_calculator``
    xTB -- ORCA's own native implementation by default, or the external xtb
    binary -- and Grimme "3c" composite methods: optimisers, Sella TS
    searches, NEB paths, conformer screens.
``orca_calculator``
    OMol25-level DFT (wB97M-V/def2-TZVPD), set up for mechanism work: saddle
    searches, IRCs, NEB-TS, frequencies. :func:`sella_ts_search` drives a
    Sella saddle search over its gradients.
``orca_gold_standard``
    Compound CCSD(T)/CBS focal-point energies, the usual recipe

        E[CCSD(T)/CBS] = E_HF(CBS) + E_MP2corr(CBS) + [E_CCSDTcorr - E_MP2corr]_small

    with an optional DFT geometry + frequency stage in front of it, on top of
    a geometry from either of the tiers above. :func:`reaction_energy`
    differences the results.
"""

import contextlib
import io
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from math import exp, sqrt
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile, OrcaTemplate
from ase.io import read
from ase.units import Hartree, kcal, mol

# --- shared helpers ----------------------------------------------------------

_DONE_RE = re.compile(r"ORCA TERMINATED NORMALLY")


def _resolve_orca(command: str | Path | None) -> str:
    """Locate the ORCA quantum-chemistry binary, refusing look-alikes.

    ``orca`` on PATH is very often the GNOME screen reader, which would happily
    launch if we passed it straight to ASE, so anything that is not clearly the
    quantum-chemistry program is rejected up front rather than executed.

    Parameters
    ----------
    command : str or pathlib.Path or None
        Explicit path to the binary. If None, ``$ASE_ORCA_COMMAND``,
        ``$ORCA_COMMAND``, ``$ORCA_PATH``/``$ORCA_DIR`` -- which may name
        either the executable itself or the directory holding it -- and PATH
        are tried in that order.

    Returns
    -------
    str
        Absolute path to the resolved binary.

    Raises
    ------
    FileNotFoundError
        If no candidate was found, or the candidate does not exist.
    RuntimeError
        If the candidate exists but does not look like ORCA.
    """
    candidate = (
        command or os.environ.get("ASE_ORCA_COMMAND") or os.environ.get("ORCA_COMMAND")
    )
    if not candidate:
        hint = os.environ.get("ORCA_PATH") or os.environ.get("ORCA_DIR")
        if hint:
            # Either the executable itself or its install directory.
            p = Path(hint).expanduser()
            candidate = p / "orca" if p.is_dir() else p
        else:
            candidate = shutil.which("orca")
    if not candidate:
        raise FileNotFoundError(
            "ORCA executable not found. Pass orca_path='/path/to/orca' or set "
            "$ORCA_PATH."
        )

    path = Path(str(candidate).split()[0]).expanduser()
    resolved = shutil.which(str(path)) or str(path)
    path = Path(resolved).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ORCA executable {path} does not exist")

    # Real ORCA ships its module binaries alongside the driver; the screen
    # reader is a lone Python script.
    siblings = any(path.parent.glob("orca_[gs]*"))
    is_elf = path.read_bytes()[:4] == b"\x7fELF"
    if not (siblings or is_elf):
        raise RuntimeError(
            f"{path} does not look like the ORCA quantum-chemistry program "
            "(no orca_* module binaries next to it, and it is not a binary). "
            "On most Linux systems /usr/bin/orca is the GNOME screen reader. "
            "Pass orca_path='/path/to/orca/orca' explicitly."
        )
    return str(path)


def _terminated_normally(out: Path) -> bool:
    """Report whether an ORCA output file records a clean exit.

    Parameters
    ----------
    out : pathlib.Path
        Path to an ``orca.out``; it need not exist.

    Returns
    -------
    bool
        True if the file is readable and contains ORCA's termination banner.
    """
    try:
        return bool(_DONE_RE.search(out.read_text(errors="replace")))
    except OSError:
        return False


# =============================================================================
# Preset calculators, ORCA-driven optimisation and GOAT conformer search
# =============================================================================


def orca_calc_preset(
    orca_path=None,
    directory=None,
    calc_type="DFT",
    xc="r2SCAN-3c",
    charge=0,
    multiplicity=1,
    basis_set="",
    n_procs=1,
    f_solv=False,
    f_disp=False,
    atom_list=None,
    calc_extra=None,
    blocks_extra=None,
    scf_option=None,
):
    """Build an ASE ORCA calculator from a small set of common presets.

    Assembles the ORCA "simple input" line and block section for one of a
    few common calculation types, so callers do not have to hand-write ORCA
    input syntax for routine DFT, MP2, CCSD(T) or QM/XTB2 jobs.

    Parameters
    ----------
    orca_path : str or None, optional
        Path to the ORCA executable; see :func:`_resolve_orca` for the
        environment fallbacks. A missing or bogus binary raises here, at
        construction, rather than at run time.
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
    if directory is None:
        directory = os.path.join(tempfile.mkdtemp(), "orca")

    profile = OrcaProfile(command=_resolve_orca(orca_path))
    inpt_procs = f"%pal nprocs {n_procs} end" if n_procs > 1 else ""

    if f_solv is not None and f_solv is not False:
        # `is True`, not truthiness: a solvent name is truthy too, and testing
        # it that way overwrote every name with WATER.
        if f_solv is True:
            f_solv = "WATER"
        inpt_solv = f'\n%CPCM SMD TRUE\n    SMDSOLVENT "{f_solv}"\nEND'
    else:
        inpt_solv = ""

    if f_disp is None or f_disp is False:
        inpt_disp = ""
    else:
        inpt_disp = "D4" if f_disp is True else f_disp

    if atom_list is not None and calc_type == "QM/XTB2":
        inpt_xtb = f"\n%QMMM QMATOMS {{{atom_list}}} END END\n"
    else:
        inpt_xtb = ""

    if blocks_extra is None:
        blocks_extra = ""

    inpt_blocks = inpt_procs + inpt_solv + blocks_extra

    if calc_type == "DFT":
        inpt_simple = f"{xc} {inpt_disp} {basis_set}"
    elif calc_type == "MP2":
        inpt_simple = f"DLPNO-{calc_type} {basis_set} {basis_set}/C"
    elif calc_type == "CCSD":
        inpt_simple = f"DLPNO-{calc_type}(T) {basis_set} {basis_set}/C"
    elif calc_type == "QM/XTB2":
        inpt_simple = f"{calc_type} {xc} {inpt_disp} {basis_set}"
        inpt_blocks = inpt_procs + inpt_solv + inpt_xtb
    else:
        inpt_simple = f"{calc_type} {basis_set}"

    # NOTE: open-shell MP2/CCSD conventionally use a UHF reference;
    # UKS is kept for all methods to preserve existing behaviour.
    if multiplicity > 1 and calc_type in ("DFT", "QM/XTB2", "MP2", "CCSD"):
        inpt_simple = "UKS " + inpt_simple

    if scf_option is not None:
        inpt_simple += " " + scf_option

    if calc_extra is not None:
        inpt_simple += " " + calc_extra

    return ORCA(
        profile=profile,
        charge=charge,
        mult=multiplicity,
        directory=directory,
        orcasimpleinput=inpt_simple + " EnGrad",
        orcablocks=inpt_blocks,
    )


# Ready-made argument sets for :func:`orca_calc_preset`, splatted into it:
# ``orca_calc_preset(**orca_preset_dft_gold)``. They name the levels of theory
# worth reaching for by habit rather than by deliberation, so that a script
# says which one it wanted instead of spelling out a functional and a basis.
# Override anything on top of a preset by passing it as a keyword after the
# splat, since later keywords win: ``orca_calc_preset(**orca_preset_dft_gold,
# n_procs=8)``.

#: Cheap DFT: BLYP/6-31+G(d,p), gas phase, no dispersion. For a first look at
#: a structure, or for the many single points of a scan.
orca_preset_dft_cheap = {
    "calc_type": "DFT",
    "xc": "BLYP",
    "basis_set": "6-31+G(d,p)",
    "f_disp": False,
    "f_solv": False,
    "atom_list": None,
    "calc_extra": None,
    "scf_option": None,
}

#: Production DFT: B3LYP/def2-SVP with D4 dispersion in implicit water.
#: ``f_disp``/``f_solv`` are ``True`` rather than named, which
#: :func:`orca_calc_preset` reads as ``D4`` and ``WATER``; pass a string
#: instead for a different solvent or dispersion correction.
orca_preset_dft_gold = {
    "calc_type": "DFT",
    "xc": "B3LYP",
    "basis_set": "DEF2-SVP",
    "f_disp": True,
    "f_solv": True,
    "atom_list": None,
    "calc_extra": None,
    "scf_option": None,
}

#: GFN2-xTB: a semi-empirical tight-binding method, orders of magnitude
#: cheaper than DFT. Fast enough to drive a NEB with, which is what makes it
#: the usual choice for a first band before refining at a higher level.
orca_preset_xtb = {
    "calc_type": "XTB2",
    "xc": "",
    "basis_set": "",
    "f_disp": False,
    "f_solv": False,
    "atom_list": None,
    "calc_extra": None,
    "scf_option": None,
}

#: DLPNO-MP2/def2-TZVPP in implicit water, for a correlated energy on a
#: geometry optimised more cheaply.
orca_preset_mp2_gold = {
    "calc_type": "MP2",
    "xc": "",
    "basis_set": "DEF2-TZVPP",
    "f_disp": False,
    "f_solv": True,
    "atom_list": None,
    "calc_extra": None,
    "scf_option": None,
}

#: CCSD(T)/def2-TZVPP in implicit water, the reference energy to judge the
#: others against. Note the ``calc_type`` is the literal ORCA keyword
#: ``'CCSD(T)'``, which :func:`orca_calc_preset` passes straight through, and
#: so runs *canonical* CCSD(T). Pass ``calc_type='CCSD'`` instead for the
#: linear-scaling ``DLPNO-CCSD(T)`` approximation, which is the only tractable
#: option beyond a handful of atoms.
orca_preset_ccsd_gold = {
    "calc_type": "CCSD(T)",
    "xc": "",
    "basis_set": "DEF2-TZVPP",
    "f_disp": False,
    "f_solv": True,
    "atom_list": None,
    "calc_extra": None,
    "scf_option": None,
}


def orca_optimise_atoms(
    atoms,
    charge=0,
    multiplicity=1,
    orca_path=None,
    xc="r2SCAN-3c",
    basis_set="",
    tight_opt=True,
    tight_scf=False,
    f_solv=False,
    f_disp=False,
    n_procs=1,
):
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
        Path to the ORCA executable; see :func:`_resolve_orca` for the
        environment fallbacks. A missing or bogus binary raises here, at
        construction, rather than at run time.
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
    opt_option = "TIGHTOPT" if tight_opt else "OPT"
    calc_extra = f"{opt_option} TIGHTSCF" if tight_scf else opt_option

    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, "orca.xyz")
        calc = orca_calc_preset(
            orca_path=orca_path,
            directory=temp_dir,
            charge=charge,
            multiplicity=multiplicity,
            xc=xc,
            basis_set=basis_set,
            n_procs=n_procs,
            f_solv=f_solv,
            f_disp=f_disp,
            calc_extra=calc_extra,
        )
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
    header_pat = re.compile(r"Conformer\s+Energy.*% total", re.IGNORECASE)
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

    return pd.DataFrame(rows, columns=["Conformer", "Energy_kcal_mol", "Percent_total"])


def orca_calculate_goat(
    atoms, charge=0, multiplicity=1, orca_path=None, n_procs=1, method="XTB"
):
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
        Path to the ORCA executable; see :func:`_resolve_orca` for the
        environment fallbacks. A missing or bogus binary raises here, at
        construction, rather than at run time.
    n_procs : int, optional
        Number of MPI processes requested via ``%pal``.
    method : str, optional
        Level of theory GOAT explores at, appended to the ``GOAT`` keyword.
        The default ``'XTB'`` is what makes the search affordable: GOAT runs
        many optimisations, so the method has to be a cheap one. Anything
        ORCA accepts as a method keyword works; ``''`` leaves the input line
        as a bare ``GOAT``, which falls back to ORCA's own default.

    Returns
    -------
    tuple
        ``(atoms, df)``: every conformer in the final ensemble, in the order
        ORCA wrote them, and their energies and populations as parsed by
        :func:`_extract_conformer_info`.
    """
    profile = OrcaProfile(command=_resolve_orca(orca_path))
    inpt_procs = f"%pal nprocs {n_procs} end" if n_procs > 1 else ""

    with tempfile.TemporaryDirectory() as temp_dir:
        calc = ORCA(
            profile=profile,
            charge=charge,
            mult=multiplicity,
            directory=temp_dir,
            orcasimpleinput=f"GOAT {method}".strip(),
            orcablocks=inpt_procs,
        )
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        xyz_file = os.path.join(temp_dir, "orca.finalensemble.xyz")
        orca_file = os.path.join(temp_dir, "orca.out")

        df = _extract_conformer_info(orca_file)
        atoms = read(xyz_file, format="xyz", index=":")
        return atoms, df


# =============================================================================
# Cheap calculators: xTB and Grimme "3c" composites
# =============================================================================


class _QuietOrcaTemplate(OrcaTemplate):
    """ASE's ORCA template with the spurious per-gradient caveat suppressed.

    ASE prints a four-line caveat every time it reads an ``orca.engrad``,
    warning that ORCA does not supply forces for the converged geometry of an
    *ORCA-internal* optimisation. We are doing ASE-driven single-point
    gradients, where that does not apply, so over a few hundred optimiser steps
    it is pure noise. Only stdout is captured -- exceptions still propagate.
    """

    def read_results(self, directory):
        """Read ORCA results as ASE would, discarding anything printed.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory holding the completed ORCA run.

        Returns
        -------
        dict
            Whatever :meth:`OrcaTemplate.read_results` returns.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            return super().read_results(directory)


#: alias -> (ORCA keyword, description), cheapest first. The GFN keywords here
#: are the external-interface ones; see :data:`NATIVE_XTB_METHODS` for their
#: in-ORCA equivalents, which is what these aliases resolve to by default.
CHEAP_METHODS: dict[str, tuple[str, str]] = {
    "gfn-ff": ("XTBFF", "GFN-FF force field, seconds per structure"),
    "gfn1-xtb": ("XTB1", "GFN1-xTB semi-empirical tight binding"),
    "gfn2-xtb": ("XTB2", "GFN2-xTB, the usual screening workhorse"),
    "hf-3c": ("HF-3c", "HF/MINIX + gCP + D3, cheapest wavefunction level"),
    "pbeh-3c": ("PBEh-3c", "PBE hybrid / def2-mSVP composite"),
    "b97-3c": ("B97-3c", "B97 GGA / def2-mTZVP composite"),
    "r2scan-3c": ("r2SCAN-3c", "r2SCAN meta-GGA / def2-mTZVPP, best 3c geometries"),
}

_XTB_METHODS = frozenset({"gfn-ff", "gfn1-xtb", "gfn2-xtb"})

#: alias -> (restricted, spin-polarised) keyword for ORCA's own xTB code. ORCA
#: 6 implements the GFN Hamiltonians internally: no external binary, MPI
#: parallel under ``%pal``, and configured with the ordinary ``%scf`` block
#: rather than a ``%xtb`` one, which it rejects outright. GFN-FF has no native
#: counterpart and stays on the external interface.
NATIVE_XTB_METHODS: dict[str, tuple[str, str]] = {
    "gfn1-xtb": ("NATIVE-XTB1", "NATIVE-SPXTB1"),
    "gfn2-xtb": ("NATIVE-XTB2", "NATIVE-SPXTB2"),
}


def _find_xtb(orca: str | Path) -> str | None:
    """Locate the driver ORCA's *external* xTB interface shells out to.

    Searches where ORCA itself does: ``$XTBEXE``, then ``otool_xtb`` and ``xtb``
    beside the ORCA binary, then ``xtb`` on PATH. ORCA 6 no longer bundles
    ``otool_xtb``, so on a stock install this finds something only if xtb was
    installed separately.

    Parameters
    ----------
    orca : str or pathlib.Path
        Resolved path to the ORCA binary, whose directory is searched.

    Returns
    -------
    str or None
        Path to the driver, or None if the external interface has nothing to
        call.
    """
    explicit = os.environ.get("XTBEXE")
    if explicit and Path(explicit).is_file():
        return explicit
    for name in ("otool_xtb", "xtb"):
        candidate = Path(orca).parent / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which("xtb")


def _method_keyword(
    key: str, native: bool | str, spin_polarised: bool
) -> tuple[str, bool]:
    """Map a :data:`CHEAP_METHODS` alias to the ORCA keyword to write.

    Parameters
    ----------
    key : str
        Alias, already lowercased and stripped, and known to be in
        :data:`CHEAP_METHODS`.
    native : bool or str
        True forces ORCA's native xTB, False forces the external interface,
        ``"auto"`` takes the native route wherever one exists. Only the xTB
        family has two routes, so this is ignored for the "3c" composites.
    spin_polarised : bool
        Use the spin-polarised native Hamiltonian. Plain xTB stays restricted
        even at ``multiplicity > 1``, filling the frontier orbitals fractionally; the
        spin-polarised variant goes unrestricted and adds the spin correction,
        which is what you want for radicals.

    Returns
    -------
    tuple of (str, bool)
        The keyword for the ``!`` line, and whether it is the native one.

    Raises
    ------
    ValueError
        If a native-only option is asked of a method that has no native
        implementation, or *native* is not True, False or ``"auto"``.
    """
    if native not in (True, False, "auto"):
        raise ValueError(f"native must be True, False or 'auto', got {native!r}")

    variants = NATIVE_XTB_METHODS.get(key)
    if native is True and variants is None and key in _XTB_METHODS:
        raise ValueError(
            f"{key!r} has no native ORCA implementation; it runs only through the "
            f"external xtb interface. Pass native=False, or pick one of "
            f"{sorted(NATIVE_XTB_METHODS)}."
        )
    use_native = variants is not None and native is not False

    if spin_polarised and not use_native:
        raise ValueError(
            f"spin_polarised=True needs ORCA's native xTB, which is available for "
            f"{sorted(NATIVE_XTB_METHODS)}; got {key!r} with native={native!r}"
        )
    if use_native:
        return variants[1] if spin_polarised else variants[0], True
    return CHEAP_METHODS[key][0], False


def _is_native_xtb(method: str) -> bool:
    """Report whether an ORCA keyword line selects the native xTB code.

    Parameters
    ----------
    method : str
        Level of theory as written on the ``!`` line, e.g. ``"NATIVE-XTB2"``.

    Returns
    -------
    bool
        True for ORCA's internal implementation, False for the external
        interface and for everything that is not xTB at all.
    """
    return any(t.startswith("native-") and "xtb" in t for t in method.lower().split())


def _carries_own_basis(method: str) -> bool:
    """Report whether an ORCA keyword line already implies a basis set.

    True for the xTB Hamiltonians, which have their own minimal basis, and for
    the "3c" composites, which are defined with one. Naming a basis alongside
    either is at best redundant and, for native xTB, fatal -- it refuses to run
    with RI, which is what the DFT default drags in.

    Parameters
    ----------
    method : str
        Level of theory as written on the ``!`` line.

    Returns
    -------
    bool
        True if no separate basis keyword should be added.
    """
    return any(
        "-3c" in t or "xtb" in t or t.startswith("gfn") for t in method.lower().split()
    )


def orca_cheap_calculator(
    method: str = "gfn2-xtb",
    directory: str | Path = "orca_cheap",
    charge: int = 0,
    multiplicity: int = 1,
    orca_path: str | Path | None = None,
    n_procs: int = 1,
    maxcore: int = 3000,
    solvent: str | None = None,
    solvent_model: str = "auto",
    forces: bool = True,
    scf_convergence: str = "",
    extra_keywords: str = "",
    extra_blocks: str = "",
    quiet: bool = True,
    native: bool | str = "auto",
    spin_polarised: bool = False,
) -> ORCA:
    """Build an ASE ORCA calculator at a cheap level of theory.

    These are the levels you drive ASE with -- optimisers, Sella TS searches,
    NEB paths, conformer screens -- to make the geometries that
    :func:`orca_gold_standard` then puts a CCSD(T)/CBS energy on.

    Parameters
    ----------
    method : str, optional
        One of :data:`CHEAP_METHODS`, cheapest first: ``gfn-ff``, ``gfn1-xtb``,
        ``gfn2-xtb``, ``hf-3c``, ``pbeh-3c``, ``b97-3c``, ``r2scan-3c``. The two
        GFN-xTB levels run through ORCA's native implementation by default;
        ``gfn-ff`` has none, so it needs an external xtb binary.
    directory : str or pathlib.Path, optional
        Working directory. **Give every NEB image, and every parallel worker, a
        separate one** -- they all write ``orca.inp``/``orca.out`` and will
        otherwise overwrite each other. Reusing one directory across the steps
        of a single optimisation is good: ORCA's AutoStart picks up the previous
        ``orca.gbw`` and converges the next SCF much faster.
    charge : int, optional
        Total charge of the system.
    multiplicity : int, optional
        Spin multiplicity, ``2S + 1``.
    orca_path : str or pathlib.Path or None, optional
        Path to the ORCA binary; see :func:`_resolve_orca` for the fallbacks.
    n_procs : int, optional
        MPI ranks for ``%pal``.
    maxcore : int, optional
        Memory per rank in MB.
    solvent : str or None, optional
        ORCA solvent name, e.g. ``"water"``. None gives gas phase.
    solvent_model : str, optional
        ``"auto"`` picks ALPB for xTB and CPCM for the DFT/HF levels; override
        with an explicit ``"CPCM"``, ``"SMD"`` or ``"ALPB"`` if you need it.
        Native xTB does run CPCM and SMD, but only by falling back to ORCA's
        general SCF machinery, which gives up much of the speed; ALPB is the
        one it solves natively.
    forces : bool, optional
        Adds ``EnGrad`` to the keyword line. Leave this on for anything that
        moves atoms: ASE's ORCA template ignores the requested properties, so
        without it ORCA never prints a gradient and ``get_forces()`` fails.
        Turn it off only for pure single-point energies.
    scf_convergence : str, optional
        SCF convergence keyword, e.g. ``"TightSCF"``. Empty uses ORCA's default.
    extra_keywords, extra_blocks : str, optional
        Appended verbatim to the ``!`` line and the ``%`` blocks respectively,
        e.g. ``extra_keywords="SlowConv"`` for a sulky SCF.
    quiet : bool, optional
        Suppress the four-line caveat ASE prints on every single gradient read.
        Set False to get stock ASE behaviour back.
    native : bool or str, optional
        Which xTB implementation to use, for the GFN-xTB levels only.
        ``"auto"`` (the default) and True take ORCA's native code, which needs
        no external binary, runs under ``%pal`` and is configured through the
        ordinary ``%scf`` block. False takes the external interface, which
        shells out to ``otool_xtb``/``xtb`` -- ORCA 6 stopped bundling that, so
        it has to be installed separately and pointed at with ``$XTBEXE``.
    spin_polarised : bool, optional
        Use the spin-polarised native Hamiltonian, ``NATIVE-SPXTB*``. Plain xTB
        stays restricted even at ``multiplicity > 1`` and fills the frontier orbitals
        fractionally; this goes unrestricted and adds the spin-polarisation
        correction, which is what open-shell radicals want. Native only.

    Returns
    -------
    ase.calculators.orca.ORCA
        A plain ASE calculator, so it composes with anything that takes one.

    Raises
    ------
    ValueError
        If *method* is not in :data:`CHEAP_METHODS`, if a native-only option is
        asked of a method with no native implementation, or if *extra_blocks*
        carries a ``%xtb`` block into a native run.
    FileNotFoundError
        If the external interface was chosen but no xtb driver could be found.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from ase.optimize import BFGS
    >>> atoms = molecule("H2O")
    >>> atoms.calc = orca_cheap_calculator("gfn2-xtb", directory="opt/mono")  # doctest: +SKIP
    >>> BFGS(atoms).run(fmax=0.02)  # doctest: +SKIP
    """
    key = method.lower().strip()
    if key not in CHEAP_METHODS:
        options = ", ".join(CHEAP_METHODS)
        raise ValueError(f"unknown cheap method {method!r}; pick one of: {options}")
    keyword, is_native = _method_keyword(key, native, spin_polarised)

    command = _resolve_orca(orca_path)
    # ORCA aborts on "%xtb ... end" next to a native keyword rather than
    # ignoring it, so catch it here where the message can say why.
    if is_native and "%xtb" in extra_blocks.lower():
        raise ValueError(
            "the %xtb block configures the external xtb interface only, and "
            "ORCA rejects it alongside a native method; pass native=False to "
            "use it, or move the setting into %scf"
        )
    if not is_native and key in _XTB_METHODS and _find_xtb(command) is None:
        alternative = (
            "pass native=True for ORCA's own implementation"
            if key in NATIVE_XTB_METHODS
            else f"{key} has no native implementation, so it needs the binary"
        )
        raise FileNotFoundError(
            f"the external xtb interface has nothing to call for {method!r}: no "
            f"driver in $XTBEXE, beside {command}, or on PATH. ORCA 6 no longer "
            f"ships otool_xtb, so install xtb and set $XTBEXE -- or {alternative}."
        )

    parts = [keyword]
    if forces:
        parts.append("EnGrad")
    if solvent:
        model = solvent_model
        if model == "auto":
            model = "ALPB" if key in _XTB_METHODS else "CPCM"
        parts.append(f"{model}({solvent})")
    if scf_convergence:
        parts.append(scf_convergence)
    if extra_keywords:
        parts.append(extra_keywords)

    blocks = f"%pal nprocs {n_procs} end\n%maxcore {maxcore}"
    if extra_blocks:
        blocks += "\n" + extra_blocks

    calc = ORCA(
        profile=OrcaProfile(command=command),
        directory=directory,
        charge=charge,
        mult=multiplicity,
        orcasimpleinput=" ".join(parts),
        orcablocks=blocks,
    )
    if quiet:
        calc.template = _QuietOrcaTemplate()
    return calc


# =============================================================================
# Mechanism calculator: OMol25-level DFT for TS search, IRC, NEB, frequencies
# =============================================================================

FUNCTIONAL = "wB97M-V"
DEFAULT_BASIS = "def2-TZVPD"

LOT_KEYWORDS: Sequence[str] = (
    "RIJCOSX",  # RI-J Coulomb + chain-of-spheres exchange
    "def2/J",  # Coulomb fitting basis
    "NoUseSym",  # no point-group symmetry
    "DEFGRID3",  # 590-point XC grid / 302-point final COSX grid
    "TightSCF",
)

# Integral thresholds: energy/force consistency, later adopted as ORCA defaults
THRESH_BLOCK = "%scf THRESH 1e-12 TCUT 1e-13 end"

NBO_BLOCK = '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end'
POP_BLOCK = (
    "%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 "
    "Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 end"
)

ECP_SIZE = {
    **{z: 28 for z in range(37, 55)},
    **{z: 46 for z in range(55, 58)},
    **{z: 28 for z in range(58, 72)},
    **{z: 60 for z in range(72, 87)},
}

TASK_KEYWORDS = {
    "sp": [],
    "engrad": ["EnGrad"],  # the only task ASE can read forces from
    "opt": ["Opt"],
    "opt+freq": ["Opt", "NumFreq"],
    "optts": ["OptTS"],
    "optts+freq": ["OptTS", "NumFreq"],
    "neb-ts": ["NEB-TS"],  # band + saddle refinement
    "freq": ["NumFreq"],  # numerical only - VV10 blocks AnFreq
    "irc": ["IRC"],
    "scan": ["Opt"],  # relaxed scan; pair with scan_coord
}


def _symmetry_breaking_block(atoms: Atoms, charge: int) -> str:
    """Build a ``%scf rotate`` block that breaks spin symmetry in a UKS singlet.

    Mixes the HOMO and LUMO by 20 degrees, which is enough to let the SCF fall
    into a broken-symmetry solution instead of the closed-shell one.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure, used only for its electron count.
    charge : int
        Total charge of the system.

    Returns
    -------
    str
        The ORCA ``%scf`` block.
    """
    n_el = sum(atoms.get_atomic_numbers()) - charge
    n_el -= sum(ECP_SIZE.get(z, 0) for z in atoms.get_atomic_numbers())
    lumo = n_el // 2
    return f"%scf rotate {{{lumo - 1}, {lumo}, 20, 1, 1}} end end"


def _solvation(solvent: str, model: str) -> tuple[list[str], list[str]]:
    """Translate a solvation request into ORCA keywords and blocks.

    Parameters
    ----------
    solvent : str
        ORCA solvent name, e.g. ``"water"``.
    model : str
        ``"cpcm"`` or ``"smd"``, case-insensitive.

    Returns
    -------
    tuple of (list of str, list of str)
        Keywords for the ``!`` line and ``%`` blocks to append.

    Raises
    ------
    ValueError
        If *model* is neither ``"cpcm"`` nor ``"smd"``.
    """
    model = model.lower()
    if model == "cpcm":
        return [f"CPCM({solvent})"], []
    if model == "smd":
        # SMD is layered on top of the CPCM electrostatics
        return [f"CPCM({solvent})"], [f'%cpcm smd true SMDsolvent "{solvent}" end']
    raise ValueError(f"solvation_model must be 'cpcm' or 'smd', got {model!r}")


def _geom_block(
    task: str,
    calc_hess: bool,
    hybrid_hess_atoms: Optional[Sequence[int]],
    recalc_hess: Optional[int],
    inhess_file: Optional[str],
    ts_mode: Optional[int],
    maxiter: Optional[int],
    scan_coord: Optional[str],
) -> str:
    """Assemble the ``%geom`` block for an ORCA-driven geometry task.

    Parameters
    ----------
    task : str
        Task name; present for symmetry with the caller and unused directly.
    calc_hess : bool
        Compute an exact Hessian before the first step. Ignored when
        *inhess_file* is given, since a stored Hessian is cheaper.
    hybrid_hess_atoms : sequence of int or None
        Atom indices to treat exactly within an otherwise model Hessian.
    recalc_hess : int or None
        Recompute the Hessian every this many steps.
    inhess_file : str or None
        Read the starting Hessian from this file instead of computing one.
    ts_mode : int or None
        Index of the mode to follow uphill in a saddle search.
    maxiter : int or None
        Cap on geometry steps.
    scan_coord : str or None
        Relaxed-scan specification, e.g. ``"B 0 5 = 1.80, 1.00, 17"`` to
        compress a bond in 17 steps.

    Returns
    -------
    str
        The ``%geom`` block, or an empty string if nothing needed setting.
    """
    parts: list[str] = []

    if inhess_file:
        parts += ["InHess Read", f'InHessName "{inhess_file}"']
    elif calc_hess:
        parts.append("Calc_Hess true")
        if hybrid_hess_atoms:
            idx = " ".join(str(i) for i in hybrid_hess_atoms)
            parts.append(f"Hybrid_Hess {{{idx}}} end")

    if recalc_hess:
        parts.append(f"Recalc_Hess {recalc_hess}")
    if ts_mode is not None:
        parts.append(f"TS_Mode {{M {ts_mode}}} end")
    if maxiter:
        parts.append(f"MaxIter {maxiter}")
    if scan_coord:
        parts.append(f"Scan {scan_coord} end")

    return "%geom " + " ".join(parts) + " end" if parts else ""


def orca_calculator(
    charge: int = 0,
    multiplicity: int = 1,
    *,
    task: str = "engrad",
    atoms: Optional[Atoms] = None,
    basis: str = DEFAULT_BASIS,
    # --- solvation ---
    solvent: Optional[str] = None,
    solvation_model: str = "cpcm",
    # --- spin ---
    open_shell_singlet: bool = False,
    # --- SCF ---
    scf_strategy: str = "default",
    scf_maxiter: int = 300,
    autostart: bool = True,
    moread: Optional[str] = None,
    # --- Hessian / TS control ---
    calc_hess: bool = True,
    hybrid_hess_atoms: Optional[Sequence[int]] = None,
    recalc_hess: Optional[int] = None,
    inhess_file: Optional[str] = None,
    ts_mode: Optional[int] = None,
    geom_maxiter: Optional[int] = None,
    scan_coord: Optional[str] = None,
    # --- NEB ---
    neb_product: Optional[str] = None,
    neb_ts_guess: Optional[str] = None,
    neb_images: int = 8,
    # --- IRC ---
    irc_hess_file: Optional[str] = None,
    irc_maxiter: int = 70,
    # --- thermochemistry ---
    temperature: float = 298.15,
    freq_increment: float = 0.005,
    # --- properties ---
    nbo: bool = False,
    population: bool = False,
    # --- resources ---
    n_procs: int = 1,
    maxcore: int = 3000,
    orca_path: Optional[str] = None,
    directory: str = ".",
) -> ORCA:
    """Build an ORCA calculator at the OMol25 level, set up for mechanism work.

    The level of theory is wB97M-V/def2-TZVPD with RIJCOSX, a dense integration
    grid and tight SCF and integral thresholds -- suitable for saddle searches,
    IRCs, NEB-TS and frequencies.

    Parameters
    ----------
    charge : int, optional
        Total charge of the system.
    multiplicity : int, optional
        Spin multiplicity, ``2S + 1``.
    task : str, optional
        One of :data:`TASK_KEYWORDS`. Use ``"engrad"`` when driving from ASE
        (Sella, NEB, LBFGS) -- it is the only task ASE can parse forces from.
        The other tasks hand control to ORCA's own optimisers; ASE will run
        them but only recover the final energy.
    atoms : ase.Atoms or None, optional
        Structure, required only when ``open_shell_singlet=True``.
    basis : str, optional
        Orbital basis set.
    solvent : str or None, optional
        ORCA solvent name, e.g. ``"water"``, ``"acetonitrile"``, ``"dmso"``,
        ``"thf"``. None gives gas phase.
    solvation_model : str, optional
        ``"cpcm"`` (fast, analytic gradients, smooth) or ``"smd"`` (better
        absolute solvation free energies, non-electrostatic terms can be less
        smooth along a reaction coordinate).
    open_shell_singlet : bool, optional
        UKS + broken-symmetry guess. Leave False for ordinary heterolytic
        proton transfer, where the electron pair stays on the donor. Set True
        for HAT/PCET, where the saddle has genuine diradical character.
    scf_strategy : str, optional
        ``"default"`` -- ORCA's DIIS -> SOSCF -> TRAH ladder. Recommended here.
        ``"omol"`` -- DIIS NOSOSCF NormalConv, byte-comparable to OMol25.
        ``"slow"`` -- adds SlowConv for stubborn cases.
    scf_maxiter : int, optional
        Cap on SCF iterations.
    autostart : bool, optional
        Let ORCA reuse ``<basename>.gbw`` from a previous run in the same
        directory as the SCF guess. Gives free orbital continuity along a
        scan or optimisation; disable if charge/multiplicity changes between
        runs, or when you want each point converged independently.
    moread : str or None, optional
        Explicit ``.gbw`` file to take the starting orbitals from.
    calc_hess : bool, optional
        Compute an exact Hessian before a saddle search. Applies to the
        ``optts`` tasks only.
    hybrid_hess_atoms : sequence of int or None, optional
        Atom indices (0-based) to treat exactly in an otherwise model
        Hessian. For proton transfer, the donor/H/acceptor triad plus any
        directly bonded heavy atoms. Cuts the numerical Hessian cost sharply.
    recalc_hess : int or None, optional
        Recompute the Hessian every this many geometry steps.
    inhess_file : str or None, optional
        Read the starting Hessian from this file instead of computing one.
    ts_mode : int or None, optional
        Index of the Hessian mode to follow uphill in a saddle search.
    geom_maxiter : int or None, optional
        Cap on geometry steps.
    scan_coord : str or None, optional
        Relaxed-scan specification for ``task="scan"``, e.g.
        ``"B 0 5 = 1.80, 1.00, 17"``.
    neb_product : str or None, optional
        Product-side XYZ file. Required for ``task="neb-ts"``.
    neb_ts_guess : str or None, optional
        Optional saddle-point guess XYZ, which speeds the band up considerably.
    neb_images : int, optional
        Number of images in the band.
    irc_hess_file : str or None, optional
        Hessian to start the IRC from; without it ORCA computes one.
    irc_maxiter : int, optional
        Cap on IRC steps per direction.
    temperature : float, optional
        Temperature in K for the thermochemistry.
    freq_increment : float, optional
        Displacement in Angstrom for the numerical Hessian.
    nbo : bool, optional
        Run NBO analysis with second-order perturbative donor-acceptor terms.
    population : bool, optional
        Print Loewdin and Mulliken populations and bond orders.
    n_procs : int, optional
        MPI ranks for ``%pal``.
    maxcore : int, optional
        Memory per rank in MB.
    orca_path : str or None, optional
        Path to the ORCA binary; see :func:`_resolve_orca` for the fallbacks.
    directory : str, optional
        Working directory for the run.

    Returns
    -------
    ase.calculators.orca.ORCA
        Calculator configured for *task*.

    Raises
    ------
    ValueError
        If *task* or *scf_strategy* is unknown, if ``open_shell_singlet`` is
        set without *atoms*, or if ``task="neb-ts"`` without *neb_product*.
    FileNotFoundError
        If no ORCA binary could be found; see :func:`_resolve_orca`.
    """
    if task not in TASK_KEYWORDS:
        raise ValueError(f"task must be one of {sorted(TASK_KEYWORDS)}, got {task!r}")

    command = _resolve_orca(orca_path)

    simple = [FUNCTIONAL, basis, *LOT_KEYWORDS, *TASK_KEYWORDS[task]]
    blocks = [THRESH_BLOCK, f"%scf MaxIter {scf_maxiter} end"]

    if scf_strategy == "omol":
        simple += ["DIIS", "NOSOSCF", "NormalConv"]
    elif scf_strategy == "slow":
        simple.append("SlowConv")
    elif scf_strategy != "default":
        raise ValueError("scf_strategy must be 'default', 'omol' or 'slow'")

    if not autostart:
        simple.append("NoAutoStart")
    if moread:
        simple.append("MORead")
        blocks.append(f'%moinp "{moread}"')

    if solvent:
        s_simple, s_blocks = _solvation(solvent, solvation_model)
        simple += s_simple
        blocks += s_blocks

    # A multiplicity of 1 is the only case where the spin state is ambiguous.
    if multiplicity == 1 and open_shell_singlet:
        if atoms is None:
            raise ValueError("atoms= is required when open_shell_singlet=True")
        simple.append("UKS")
        blocks.append(_symmetry_breaking_block(atoms, charge))

    wants_geom = task in {"opt", "opt+freq", "optts", "optts+freq", "neb-ts", "scan"}
    if wants_geom:
        # Only a saddle search justifies the cost of an exact starting Hessian.
        needs_hess = task in {"optts", "optts+freq"}
        geom = _geom_block(
            task=task,
            calc_hess=calc_hess and needs_hess,
            hybrid_hess_atoms=hybrid_hess_atoms,
            recalc_hess=recalc_hess,
            inhess_file=inhess_file,
            ts_mode=ts_mode,
            maxiter=geom_maxiter,
            scan_coord=scan_coord,
        )
        if geom:
            blocks.append(geom)

    if task == "neb-ts":
        if not neb_product:
            raise ValueError("neb_product=<product.xyz> is required for task='neb-ts'")
        neb = [f'NEB_End_XYZFile "{neb_product}"', f"NImages {neb_images}"]
        if neb_ts_guess:
            neb.append(f'NEB_TS_XYZFile "{neb_ts_guess}"')
        blocks.append("%neb " + " ".join(neb) + " end")

    if task == "irc":
        irc = [f"MaxIter {irc_maxiter}", "Direction both"]
        if irc_hess_file:
            irc += ["InitHess read", f'Hess_Filename "{irc_hess_file}"']
        blocks.append("%irc " + " ".join(irc) + " end")

    # Numerical only: VV10 has no analytic second derivatives.
    if "NumFreq" in TASK_KEYWORDS[task]:
        blocks.append(
            f"%freq Temp {temperature} CentralDiff true "
            f"Increment {freq_increment} Restart true end"
        )

    if population or nbo:
        simple.append("ALLPOP")
        blocks.append(POP_BLOCK)
    if nbo:
        blocks.append(NBO_BLOCK)
    else:
        # NBO is on by default in recent ORCA and is not free.
        simple += ["NONBO", "NONPA"]

    blocks += [f"%maxcore {maxcore}", f"%pal nprocs {n_procs} end"]

    return ORCA(
        profile=OrcaProfile(command=command),
        directory=directory,
        charge=charge,
        mult=multiplicity,
        orcasimpleinput=" ".join(simple),
        orcablocks=" ".join(b for b in blocks if b),
    )


def sella_ts_search(
    atoms: Atoms,
    charge: int = 0,
    multiplicity: int = 1,
    *,
    fmax: float = 0.02,
    steps: int = 200,
    trajectory: Optional[str] = "ts_search.traj",
    **calc_kwargs,
):
    """Locate a saddle point with Sella, using ORCA only for energy+gradient.

    Sella accumulates curvature from the gradients it already needs, so this
    avoids the 6N-gradient numerical Hessian that ORCA's OptTS would trigger.
    Confirm the result afterwards with ``task="freq"``.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting guess for the saddle point. Modified in place.
    charge : int, optional
        Total charge of the system.
    multiplicity : int, optional
        Spin multiplicity, ``2S + 1``.
    fmax : float, optional
        Force convergence threshold in eV/Angstrom.
    steps : int, optional
        Maximum optimiser steps.
    trajectory : str or None, optional
        Path to write the search trajectory to; None writes nothing.
    **calc_kwargs
        Passed to :func:`orca_calculator`. ``task`` and ``atoms`` default to
        ``"engrad"`` and *atoms*.

    Returns
    -------
    ase.Atoms
        The same object, at the located saddle point.
    """
    from sella import Sella

    calc_kwargs.setdefault("task", "engrad")
    calc_kwargs.setdefault("atoms", atoms)
    atoms.calc = orca_calculator(
        charge=charge, multiplicity=multiplicity, **calc_kwargs
    )

    opt = Sella(atoms, order=1, internal=True, trajectory=trajectory)
    opt.run(fmax=fmax, steps=steps)
    return atoms


# =============================================================================
# Gold standard: compound CCSD(T)/CBS focal-point energies
# =============================================================================

EH_TO_KCAL = Hartree / (kcal / mol)

_CARDINAL_LETTER = {2: "D", 3: "T", 4: "Q", 5: "5"}
_DEF2_BASIS = {2: "def2-SVP", 3: "def2-TZVPP", 4: "def2-QZVPP"}

# (alpha, beta) for E_SCF(X) = E_CBS + A*exp(-alpha*sqrt(X)) and
# E_corr(X) = E_CBS + B*X**(-beta).  Values as tabulated in the ORCA manual.
_CBS_PARAMS: dict[str, dict[tuple[int, int], tuple[float, float]]] = {
    "cc": {(2, 3): (4.42, 2.46), (3, 4): (5.46, 3.05), (4, 5): (9.19, 3.00)},
    "aug-cc": {(2, 3): (4.30, 2.51), (3, 4): (5.79, 3.05), (4, 5): (9.19, 3.00)},
    "def2": {(2, 3): (10.39, 2.40), (3, 4): (7.88, 2.97)},
}

_PATTERNS: dict[str, str] = {
    "scf": r"Total Energy\s*:\s*(-?\d+\.\d+)\s*Eh",
    "mp2_corr": r"(?<!SCS-)MP2 CORRELATION ENERGY\s*:?\s*(-?\d+\.\d+)",
    "cc_corr": r"E\(CORR\)(?:\(total\))?[\s.:=]*(-?\d+\.\d+)",
    "final_corr": r"Final correlation energy[\s.:=]*(-?\d+\.\d+)",
    "triples": r"Triples Correction \(T\)[\s.:=]*(-?\d+\.\d+)",
    "e_ccsd": r"E\(CCSD\)[\s.:=]*(-?\d+\.\d+)",
    "e_ccsdt": r"E\(CCSD\(T\)\)[\s.:=]*(-?\d+\.\d+)",
    "final_sp": r"FINAL SINGLE POINT ENERGY\s*(-?\d+\.\d+)",
    "zpe": r"Zero point energy[\s.:=]*(-?\d+\.\d+)",
    "thermal_eel": r"Electronic energy[\s.:=]*(-?\d+\.\d+)",
    "enthalpy": r"Total Enthalpy[\s.:=]*(-?\d+\.\d+)",
    "gibbs": r"Final Gibbs free energy[\s.:=]*(-?\d+\.\d+)",
    "gibbs_corr": r"G-E\(el\)[\s.:=]*(-?\d+\.\d+)",
}
_COMPILED = {k: re.compile(v) for k, v in _PATTERNS.items()}
_IMAG_RE = re.compile(r"^\s*\d+:\s+(-\d+\.\d+)\s+cm\*\*-1\s+\*\*\*imaginary", re.M)


def _parse_orca(text: str) -> dict[str, float]:
    """Pull every recognised energy out of an ORCA output.

    Parameters
    ----------
    text : str
        Full contents of an ``orca.out``.

    Returns
    -------
    dict of {str: float}
        Keys from :data:`_PATTERNS`, values in Hartree. The last match in the
        file wins, so a compound job reports its final stage.
    """
    vals: dict[str, float] = {}
    for key, rx in _COMPILED.items():
        matches = rx.findall(text)
        if matches:
            vals[key] = float(matches[-1])
    return vals


def _correlation_energy(vals: dict[str, float]) -> float:
    """Extract the total correlation energy of the highest method present.

    ORCA reports correlation energies differently depending on the module, so
    the equivalent routes are tried in descending order of directness.

    Parameters
    ----------
    vals : dict of {str: float}
        Parsed energies, as returned by :func:`_parse_orca`.

    Returns
    -------
    float
        Correlation energy in Hartree.

    Raises
    ------
    ValueError
        If none of the recognised routes are present.
    """
    if "e_ccsdt" in vals and "scf" in vals:
        return vals["e_ccsdt"] - vals["scf"]
    if "final_corr" in vals:
        return vals["final_corr"]
    if "cc_corr" in vals:
        return vals["cc_corr"] + vals.get("triples", 0.0)
    if "mp2_corr" in vals:
        return vals["mp2_corr"]
    if "final_sp" in vals and "scf" in vals:
        return vals["final_sp"] - vals["scf"]
    raise ValueError("no correlation energy found in ORCA output")


def _extrapolate_scf(
    e_lo: float, e_hi: float, n_lo: int, n_hi: int, alpha: float
) -> float:
    """Extrapolate an SCF energy to the basis-set limit.

    Solves ``E(X) = E_CBS + A*exp(-alpha*sqrt(X))`` for the two given points.

    Parameters
    ----------
    e_lo, e_hi : float
        SCF energies in Hartree at the smaller and larger basis.
    n_lo, n_hi : int
        Cardinal numbers of those two bases.
    alpha : float
        Family-specific exponent from :data:`_CBS_PARAMS`.

    Returns
    -------
    float
        Extrapolated SCF energy in Hartree.
    """
    a, b = exp(-alpha * sqrt(n_lo)), exp(-alpha * sqrt(n_hi))
    return (e_lo * b - e_hi * a) / (b - a)


def _extrapolate_corr(
    e_lo: float, e_hi: float, n_lo: int, n_hi: int, beta: float
) -> float:
    """Extrapolate a correlation energy to the basis-set limit.

    Solves ``E(X) = E_CBS + B*X**(-beta)`` for the two given points. The
    correlation energy converges much more slowly than the SCF energy, hence
    the inverse-power rather than exponential form.

    Parameters
    ----------
    e_lo, e_hi : float
        Correlation energies in Hartree at the smaller and larger basis.
    n_lo, n_hi : int
        Cardinal numbers of those two bases.
    beta : float
        Family-specific exponent from :data:`_CBS_PARAMS`.

    Returns
    -------
    float
        Extrapolated correlation energy in Hartree.
    """
    x_lo, x_hi = n_lo**beta, n_hi**beta
    return (x_lo * e_lo - x_hi * e_hi) / (x_lo - x_hi)


def _basis_name(family: str, n: int) -> str:
    """Build the ORCA basis-set name for a family and cardinal number.

    Parameters
    ----------
    family : str
        ``"cc"``, ``"aug-cc"`` or ``"def2"``.
    n : int
        Cardinal number, 2 for double-zeta up to 5.

    Returns
    -------
    str
        Basis-set name, e.g. ``"cc-pVTZ"``.

    Raises
    ------
    ValueError
        If the family is unknown or has no basis at that cardinal number.
    """
    if family == "def2":
        if n not in _DEF2_BASIS:
            raise ValueError(f"no def2 basis for cardinal number {n}")
        return _DEF2_BASIS[n]
    if family not in _CBS_PARAMS:
        raise ValueError(
            f"unknown basis family {family!r}; pick one of {sorted(_CBS_PARAMS)}"
        )
    if n not in _CARDINAL_LETTER:
        raise ValueError(f"no {family} basis for cardinal number {n}")
    return f"{family}-pV{_CARDINAL_LETTER[n]}Z"


def _cbs_params(family: str, cardinals: tuple[int, int]) -> tuple[float, float]:
    """Look up the extrapolation exponents for a basis pair.

    Parameters
    ----------
    family : str
        Basis family, a key of :data:`_CBS_PARAMS`.
    cardinals : tuple of int
        The two cardinal numbers, smaller first.

    Returns
    -------
    tuple of float
        ``(alpha, beta)`` for :func:`_extrapolate_scf` and
        :func:`_extrapolate_corr`.

    Raises
    ------
    ValueError
        If that family and pair are not tabulated.
    """
    table = _CBS_PARAMS[family]
    if cardinals not in table:
        raise ValueError(
            f"no extrapolation parameters for {family} {cardinals}; "
            f"available: {sorted(table)}"
        )
    return table[cardinals]


def _geometry_keywords(
    opt_method: str,
    opt_basis: str | None,
    optimise: bool,
    frequencies: bool,
    transition_state: bool,
    common: str,
) -> str:
    """Build the ``!`` line for the geometry / thermochemistry stage.

    Parameters
    ----------
    opt_method : str
        Level of theory. A :data:`CHEAP_METHODS` alias is translated to its
        ORCA keyword, preferring native xTB; anything else is passed through
        as written.
    opt_basis : str or None
        Basis for a method that does not carry one. None gives def2-TZVP.
    optimise : bool
        Relax the geometry rather than take it as given.
    frequencies : bool
        Add a Hessian for the ZPE and thermal corrections.
    transition_state : bool
        Optimise to a saddle rather than a minimum.
    common : str
        Keywords shared by every stage, e.g. ``"TightSCF CPCM(water)"``.

    Returns
    -------
    str
        The keyword line.
    """
    key = opt_method.lower().strip()
    if key in CHEAP_METHODS:
        opt_method, _ = _method_keyword(key, native="auto", spin_polarised=False)

    keywords = [opt_method]
    if not _carries_own_basis(opt_method):
        keywords += [opt_basis or "def2-TZVP", "RIJCOSX", "def2/J"]
    if optimise:
        keywords.append("OptTS" if transition_state else "TightOpt")
    if frequencies:
        # Native xTB runs at a finite electronic temperature, which rules out
        # the analytic Hessian; ORCA aborts on ! Freq rather than fall back.
        keywords.append("NumFreq" if _is_native_xtb(opt_method) else "Freq")
    keywords.append(common)
    return " ".join(keywords)


@dataclass
class GoldStandard:
    """Result of an :func:`orca_gold_standard` compound job.

    Attributes
    ----------
    atoms : ase.Atoms
        Geometry the energies refer to, updated if the job optimised it.
    charge : int
        Total charge of the system.
    multiplicity : int
        Spin multiplicity, ``2S + 1``.
    e_hf_cbs : float
        Hartree-Fock energy at the basis-set limit, in Hartree.
    e_corr_cbs : float
        Total correlation energy at the basis-set limit, in Hartree.
    e_total : float
        CCSD(T)/CBS electronic energy, the sum of the two above, in Hartree.
    e_mp2_corr_cbs : float or None
        MP2 correlation energy at the limit. None on the ``extrapolate_cc``
        route, which has no MP2 stage.
    delta_cc : float or None
        The ``CCSD(T) - MP2`` correction, in Hartree. None as above.
    zpe : float or None
        Zero-point energy in Hartree, if frequencies were run.
    gibbs_correction, enthalpy_correction : float or None
        Thermal corrections ``G - E(el)`` and ``H - E(el)``, in Hartree.
    imaginary_frequencies : list of float
        Imaginary modes in cm^-1: one is expected at a saddle, none at a
        minimum.
    levels : dict of {str: str}
        Human-readable description of the level of theory per stage.
    components : dict of {str: dict of {str: float}}
        Every energy parsed, keyed by stage name.
    directories : dict of {str: pathlib.Path}
        Where each stage ran, for inspecting the raw output.
    """

    atoms: Atoms
    charge: int = 0
    multiplicity: int = 1
    e_hf_cbs: float = 0.0
    e_corr_cbs: float = 0.0
    e_total: float = 0.0
    e_mp2_corr_cbs: float | None = None
    delta_cc: float | None = None
    zpe: float | None = None
    gibbs_correction: float | None = None
    enthalpy_correction: float | None = None
    imaginary_frequencies: list[float] = field(default_factory=list)
    levels: dict[str, str] = field(default_factory=dict)
    components: dict[str, dict[str, float]] = field(default_factory=dict)
    directories: dict[str, Path] = field(default_factory=dict)

    @property
    def energy(self) -> float:
        """Electronic CCSD(T)/CBS energy in eV, i.e. the ASE convention."""
        return self.e_total * Hartree

    @property
    def enthalpy(self) -> float | None:
        """Enthalpy in Hartree, or None if no frequencies were run."""
        if self.enthalpy_correction is None:
            return None
        return self.e_total + self.enthalpy_correction

    @property
    def gibbs(self) -> float | None:
        """Gibbs energy in Hartree, or None if no frequencies were run."""
        if self.gibbs_correction is None:
            return None
        return self.e_total + self.gibbs_correction

    def summary(self) -> str:
        """Format the energy breakdown as a human-readable report.

        Returns
        -------
        str
            Multi-line table of every component that was computed, in both
            Hartree and kcal/mol, with any imaginary modes listed at the end.
        """
        rows = [
            ("HF/CBS", self.e_hf_cbs),
            ("MP2 corr/CBS", self.e_mp2_corr_cbs),
            ("delta CCSD(T)-MP2", self.delta_cc),
            ("correlation/CBS", self.e_corr_cbs),
            ("CCSD(T)/CBS", self.e_total),
            ("ZPE", self.zpe),
            ("H - E(el)", self.enthalpy_correction),
            ("G - E(el)", self.gibbs_correction),
            ("H(CCSD(T)/CBS)", self.enthalpy),
            ("G(CCSD(T)/CBS)", self.gibbs),
        ]
        lines = [
            f"{self.atoms.get_chemical_formula()}  charge={self.charge} multiplicity={self.multiplicity}"
        ]
        lines += [f"  {name}" for name in self.levels.values()]
        lines += [
            f"  {name:<20s}{value:20.8f} Eh{value * EH_TO_KCAL:15.3f} kcal/mol"
            for name, value in rows
            if value is not None
        ]
        if self.imaginary_frequencies:
            imag = ", ".join(f"{f:.1f}" for f in self.imaginary_frequencies)
            lines.append(f"  imaginary modes: {imag} cm^-1")
        return "\n".join(lines)


def orca_gold_standard(
    atoms: Atoms,
    directory: str | Path = "orca_gold",
    charge: int = 0,
    multiplicity: int = 1,
    orca_path: str | Path | None = None,
    n_procs: int = 1,
    maxcore: int = 3000,
    optimise: bool = True,
    frequencies: bool = True,
    transition_state: bool = False,
    opt_method: str = "r2SCAN-3c",
    opt_basis: str | None = None,
    cc_method: str = "DLPNO-CCSD(T)",
    cc_extra: str = "TightPNO",
    mp2_method: str = "RI-MP2",
    basis_family: str = "cc",
    cardinals: tuple[int, int] = (3, 4),
    cc_cardinal: int | None = None,
    extrapolate_cc: bool = False,
    aux_basis: str | None = None,
    scf_convergence: str = "TightSCF",
    frozen_core: bool = True,
    solvent: str | None = None,
    reuse: bool = True,
    verbose: bool = True,
) -> GoldStandard:
    """Run a compound ORCA job for a CCSD(T)/CBS energy.

    Stages, each in its own subdirectory of *directory* and each a separate ASE
    ORCA calculation:

    1. ``opt``   -- geometry optimisation (+ frequencies) at *opt_method*.
    2. ``mp2_*`` -- MP2 single points at the two *cardinals*, giving both the
       HF and the MP2 correlation energies used for the CBS limits.
    3. ``cc_*``  -- CCSD(T) single point at *cc_cardinal* for the
       CCSD(T) - MP2 correction.

    With ``extrapolate_cc=True`` the MP2 stages are skipped and the CCSD(T)
    correlation energy is extrapolated directly from both cardinals -- the most
    accurate and by far the most expensive route.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry. Copied, so the caller's object is untouched.
    directory : str or pathlib.Path, optional
        Root directory; each stage gets a subdirectory of it.
    charge : int, optional
        Total charge of the system.
    multiplicity : int, optional
        Spin multiplicity, ``2S + 1``.
    orca_path : str or pathlib.Path or None, optional
        Full path to the ORCA binary. Required for ``n_procs > 1``. Falls back to
        ``$ASE_ORCA_COMMAND``, ``$ORCA_COMMAND``, ``$ORCA_PATH`` then PATH.
    n_procs : int, optional
        MPI ranks for ``%pal``.
    maxcore : int, optional
        Memory per rank in MB.
    optimise : bool, optional
        Optimise the geometry before the single points.
    frequencies : bool, optional
        Run frequencies, giving the ZPE and thermal corrections, and verifying
        the stationary point.
    transition_state : bool, optional
        Search for a saddle rather than a minimum, and expect exactly one
        imaginary mode.
    opt_method : str, optional
        Level of theory for the geometry stage, either an ORCA keyword or a
        :data:`CHEAP_METHODS` alias -- ``"gfn2-xtb"`` resolves to the native
        implementation. A "3c" composite or an xTB Hamiltonian carries its own
        basis; anything else is paired with *opt_basis*.
    opt_basis : str or None, optional
        Basis for an *opt_method* that has none of its own. Defaults to
        def2-TZVP.
    cc_method : str, optional
        Coupled-cluster method, e.g. ``"DLPNO-CCSD(T)"``.
    cc_extra : str, optional
        Extra keywords for the CC stage. ``"TightPNO"`` keeps the local
        approximation error well below the extrapolation error.
    mp2_method : str, optional
        MP2 method for the CBS stages.
    basis_family : str, optional
        ``"cc"``, ``"aug-cc"`` or ``"def2"``. Diffuse functions matter for
        anions and for weakly bound complexes.
    cardinals : tuple of int, optional
        Basis cardinal numbers to extrapolate between, e.g. ``(3, 4)`` for
        cc-pVTZ/cc-pVQZ. ``(2, 3)`` is the cheap variant.
    cc_cardinal : int or None, optional
        Cardinal number for the coupled-cluster stage. Defaults to the smaller
        of *cardinals*; if it is not one of them an extra MP2 job is run there
        so the delta stays consistent.
    extrapolate_cc : bool, optional
        Extrapolate CCSD(T) directly instead of correcting an MP2 limit.
    aux_basis : str or None, optional
        Auxiliary basis for the RI stages; ORCA picks one automatically if
        this is None.
    scf_convergence : str, optional
        SCF convergence keyword applied to every stage.
    frozen_core : bool, optional
        Freeze core orbitals in the correlated stages.
    solvent : str or None, optional
        ORCA solvent name for a CPCM correction. None gives gas phase.
    reuse : bool, optional
        Skip stages whose output already says ``ORCA TERMINATED NORMALLY``.
    verbose : bool, optional
        Print each stage as it runs, then the final summary.

    Returns
    -------
    GoldStandard
        Energies in Hartree, plus per-stage components and directories.
    """
    n_lo, n_hi = sorted(cardinals)
    alpha, beta = _cbs_params(basis_family, (n_lo, n_hi))
    n_cc = cc_cardinal if cc_cardinal is not None else n_lo

    root = Path(directory)
    profile = OrcaProfile(command=_resolve_orca(orca_path))
    if n_procs > 1 and not Path(profile.command).is_absolute():
        raise ValueError("parallel ORCA needs an absolute path to the binary")

    common = scf_convergence
    if solvent:
        common += f" CPCM({solvent})"
    # Frozen core only means anything to the correlated stages.
    correlated = f"{common} " + ("FrozenCore" if frozen_core else "NoFrozenCore")
    blocks = f"%pal nprocs {n_procs} end\n%maxcore {maxcore}"

    result = GoldStandard(atoms=atoms.copy(), charge=charge, multiplicity=multiplicity)

    def run(name: str, simpleinput: str, extra_blocks: str = "") -> dict[str, float]:
        """Run one stage of the compound job, or reuse a finished one.

        Parameters
        ----------
        name : str
            Stage name, used as the subdirectory and the components key.
        simpleinput : str
            Contents of the ORCA ``!`` line.
        extra_blocks : str, optional
            Appended to the shared ``%`` blocks.

        Returns
        -------
        dict of {str: float}
            Energies parsed from the stage output.
        """
        workdir = root / name
        out = workdir / "orca.out"
        result.directories[name] = workdir
        if reuse and _terminated_normally(out):
            if verbose:
                print(f"  [{name}] reusing {out}")
            vals = _parse_orca(out.read_text(errors="replace"))
        else:
            workdir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"  [{name}] ! {simpleinput}")
            work = result.atoms.copy()
            work.calc = ORCA(
                profile=profile,
                directory=workdir,
                charge=charge,
                mult=multiplicity,
                orcasimpleinput=simpleinput,
                orcablocks=blocks + extra_blocks,
            )
            try:
                work.get_potential_energy()
            except Exception:
                # Frequency-only and composite jobs can defeat ASE's reader;
                # the output is still authoritative if ORCA finished.
                if not _terminated_normally(out):
                    raise
            vals = _parse_orca(out.read_text(errors="replace"))
        result.components[name] = vals
        return vals

    # ---- stage 1: geometry and thermochemistry -----------------------------
    if optimise or frequencies:
        level = _geometry_keywords(
            opt_method=opt_method,
            opt_basis=opt_basis,
            optimise=optimise,
            frequencies=frequencies,
            transition_state=transition_state,
            common=common,
        )
        result.levels["geometry"] = f"geometry / thermochemistry: {level}"

        extra = "\n%geom Calc_Hess true end" if (optimise and transition_state) else ""
        vals = run("opt", level, extra)

        if optimise:
            final_xyz = result.directories["opt"] / "orca.xyz"
            if final_xyz.is_file():
                result.atoms.set_positions(read(final_xyz).get_positions())
            elif verbose:
                print("  [opt] warning: no orca.xyz, keeping input geometry")

        text = (result.directories["opt"] / "orca.out").read_text(errors="replace")
        result.imaginary_frequencies = [float(f) for f in _IMAG_RE.findall(text)]
        result.zpe = vals.get("zpe")
        result.gibbs_correction = vals.get("gibbs_corr")
        if "enthalpy" in vals and "thermal_eel" in vals:
            result.enthalpy_correction = vals["enthalpy"] - vals["thermal_eel"]

        expected = 1 if transition_state else 0
        if frequencies and len(result.imaginary_frequencies) != expected:
            kind = "transition state" if transition_state else "minimum"
            print(
                f"  [opt] warning: {len(result.imaginary_frequencies)} imaginary "
                f"mode(s), expected {expected} for a {kind}"
            )

    # ---- stage 2: correlation stack ----------------------------------------
    def sp_level(method: str, n: int, extra: str = "") -> str:
        """Build the ``!`` line for a correlated single point.

        Parameters
        ----------
        method : str
            Correlation method keyword.
        n : int
            Basis cardinal number.
        extra : str, optional
            Extra keywords, e.g. ``"TightPNO"``.

        Returns
        -------
        str
            The keyword line.
        """
        basis = _basis_name(basis_family, n)
        parts = [method, basis]
        if aux_basis:
            parts.append(aux_basis)
        if extra:
            parts.append(extra)
        parts.append(correlated)
        return " ".join(parts)

    if extrapolate_cc:
        lo = run(f"cc_{n_lo}", sp_level(cc_method, n_lo, cc_extra))
        hi = run(f"cc_{n_hi}", sp_level(cc_method, n_hi, cc_extra))
        result.levels["correlation"] = (
            f"correlation: {cc_method}/CBS({n_lo},{n_hi}) "
            f"[{_basis_name(basis_family, n_lo)} -> {_basis_name(basis_family, n_hi)}]"
        )
        result.e_hf_cbs = _extrapolate_scf(lo["scf"], hi["scf"], n_lo, n_hi, alpha)
        result.e_corr_cbs = _extrapolate_corr(
            _correlation_energy(lo), _correlation_energy(hi), n_lo, n_hi, beta
        )
    else:
        lo = run(f"mp2_{n_lo}", sp_level(mp2_method, n_lo))
        hi = run(f"mp2_{n_hi}", sp_level(mp2_method, n_hi))
        cc = run(f"cc_{n_cc}", sp_level(cc_method, n_cc, cc_extra))

        if n_cc == n_lo:
            mp2_ref = lo["mp2_corr"]
        elif n_cc == n_hi:
            mp2_ref = hi["mp2_corr"]
        else:
            mp2_ref = run(f"mp2_{n_cc}", sp_level(mp2_method, n_cc))["mp2_corr"]

        result.levels["correlation"] = (
            f"correlation: {mp2_method}/CBS({n_lo},{n_hi}) + "
            f"[{cc_method} - {mp2_method}]/{_basis_name(basis_family, n_cc)}"
        )
        result.e_hf_cbs = _extrapolate_scf(lo["scf"], hi["scf"], n_lo, n_hi, alpha)
        result.e_mp2_corr_cbs = _extrapolate_corr(
            lo["mp2_corr"], hi["mp2_corr"], n_lo, n_hi, beta
        )
        result.delta_cc = _correlation_energy(cc) - mp2_ref
        result.e_corr_cbs = result.e_mp2_corr_cbs + result.delta_cc

    result.e_total = result.e_hf_cbs + result.e_corr_cbs
    if verbose:
        print(result.summary())
    return result


def reaction_energy(
    reactants: list[GoldStandard],
    products: list[GoldStandard],
    thermo: str = "gibbs",
) -> float:
    """Compute a reaction energy from completed compound jobs.

    Parameters
    ----------
    reactants, products : list of GoldStandard
        The two sides of the reaction. List a species once per equivalent, so
        the stoichiometry balances.
    thermo : str, optional
        Which energy to difference: ``"electronic"``, ``"enthalpy"`` or
        ``"gibbs"``. The latter two need jobs run with ``frequencies=True``.

    Returns
    -------
    float
        Reaction energy in kcal/mol.

    Raises
    ------
    KeyError
        If *thermo* is not one of the three names above.
    ValueError
        If any species lacks the requested thermal correction.
    """
    attr = {"electronic": "e_total", "enthalpy": "enthalpy", "gibbs": "gibbs"}[thermo]

    def total(species: list[GoldStandard]) -> float:
        """Sum the requested energy over one side of the reaction."""
        values = [getattr(s, attr) for s in species]
        if any(v is None for v in values):
            raise ValueError(f"{thermo} energy missing; run with frequencies=True")
        return sum(values)

    return (total(products) - total(reactants)) * EH_TO_KCAL
