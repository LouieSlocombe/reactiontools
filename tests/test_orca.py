"""Tests for reactiontools.tools_orca.

Unlike the rest of the suite, these need a real ORCA install: the functions
shell out to the binary rather than wrapping a Python calculator, so there is
nothing meaningful left to test without it. They are skipped unless
``ORCA_PATH`` points at the executable.

The reference energies are for ``tests/data/fad.xyz`` at the default
``r2SCAN-3c`` level, and are tied to both that geometry and the ORCA version
that produced them.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactiontools import (orca_calc_preset,
                           orca_calculate_goat,
                           orca_optimise_atoms,
                           orca_preset_ccsd_gold,
                           orca_preset_dft_cheap,
                           orca_preset_dft_gold,
                           orca_preset_mp2_gold,
                           orca_preset_xtb)

# Skip rather than fail: ORCA is licensed separately and installed by hand, so
# a machine without it is the normal case, not a broken one.
orca_required = pytest.mark.skipif(
    os.environ.get("ORCA_PATH") is None,
    reason="needs ORCA; set ORCA_PATH to the executable")

DATA = Path(__file__).parent / "data"


@pytest.fixture
def fad():
    """Formic acid dimer, the reference geometry for the ORCA tests."""
    return read(DATA / "fad.xyz")


@orca_required
def test_orca_calc_preset(fad):
    fad.calc = orca_calc_preset()
    energy = fad.get_potential_energy()

    assert np.allclose(energy, -10325.045291755621)


@orca_required
def test_orca_optimise_atoms(fad):
    opt_atoms = orca_optimise_atoms(fad)
    opt_atoms.calc = orca_calc_preset()
    energy = opt_atoms.get_potential_energy()

    # Relaxing can only lower the energy from the single point above.
    assert np.allclose(energy, -10326.977956847948)


@orca_required
def test_orca_calculate_goat(fad):
    conformers, df = orca_calculate_goat(fad)

    assert len(conformers) == len(df)
    assert list(df.columns) == ["Conformer", "Energy_kcal_mol", "Percent_total"]


def test_orca_calc_preset_builds_the_simple_input_line():
    """The input assembly is pure string work, so it runs without ORCA."""
    calc = orca_calc_preset(orca_path="/nonexistent/orca",
                            xc="PBE0",
                            basis_set="def2-TZVP",
                            f_disp=True,
                            f_solv="TOLUENE",
                            n_procs=4)

    simple = calc.parameters["orcasimpleinput"]
    blocks = calc.parameters["orcablocks"]

    assert simple.split() == ["PBE0", "D4", "def2-TZVP", "EnGrad"]
    assert "%pal nprocs 4 end" in blocks
    assert 'SMDSOLVENT "TOLUENE"' in blocks


def test_orca_calc_preset_defaults_solvation_and_dispersion_keywords():
    """``True`` picks the default; a string is passed through as given."""
    default = orca_calc_preset(orca_path="/nonexistent/orca",
                               f_solv=True, f_disp=True)

    assert 'SMDSOLVENT "WATER"' in default.parameters["orcablocks"]
    assert "D4" in default.parameters["orcasimpleinput"]

    named = orca_calc_preset(orca_path="/nonexistent/orca", f_disp="D3BJ")

    assert "D3BJ" in named.parameters["orcasimpleinput"]


def test_orca_calc_preset_omits_solvation_and_dispersion_by_default():
    calc = orca_calc_preset(orca_path="/nonexistent/orca")

    assert "SMD" not in calc.parameters["orcablocks"]
    assert "D4" not in calc.parameters["orcasimpleinput"]


def test_orca_calc_preset_uses_an_unrestricted_reference_when_open_shell():
    calc = orca_calc_preset(orca_path="/nonexistent/orca", multiplicity=3)

    assert calc.parameters["orcasimpleinput"].startswith("UKS ")


def test_orca_calc_preset_builds_a_qmmm_region():
    calc = orca_calc_preset(orca_path="/nonexistent/orca",
                            calc_type="QM/XTB2",
                            atom_list="0:5")

    assert calc.parameters["orcasimpleinput"].startswith("QM/XTB2 ")
    assert "%QMMM QMATOMS {0:5} END END" in calc.parameters["orcablocks"]


class TestPresets:
    """The ``orca_preset_*`` dictionaries, splatted into orca_calc_preset.

    Like the tests above these are pure string assembly, so they run without
    ORCA. They pin the level of theory each preset names, which is the whole
    point of having them: a preset that quietly changed functional would
    change every result taken with it.
    """

    def test_dft_cheap_is_blyp_in_the_gas_phase(self):
        calc = orca_calc_preset(orca_path="/nonexistent/orca",
                                **orca_preset_dft_cheap)

        assert calc.parameters["orcasimpleinput"].split() == [
            "BLYP", "6-31+G(d,p)", "EnGrad"]
        assert "SMD" not in calc.parameters["orcablocks"]

    def test_dft_gold_is_b3lyp_with_d4_in_water(self):
        calc = orca_calc_preset(orca_path="/nonexistent/orca",
                                **orca_preset_dft_gold)

        assert calc.parameters["orcasimpleinput"].split() == [
            "B3LYP", "D4", "DEF2-SVP", "EnGrad"]
        assert 'SMDSOLVENT "WATER"' in calc.parameters["orcablocks"]

    def test_xtb_names_the_method_and_takes_no_basis(self):
        calc = orca_calc_preset(orca_path="/nonexistent/orca",
                                **orca_preset_xtb)

        assert calc.parameters["orcasimpleinput"].split() == ["XTB2", "EnGrad"]

    def test_mp2_gold_uses_the_dlpno_approximation_and_an_aux_basis(self):
        calc = orca_calc_preset(orca_path="/nonexistent/orca",
                                **orca_preset_mp2_gold)

        assert calc.parameters["orcasimpleinput"].split() == [
            "DLPNO-MP2", "DEF2-TZVPP", "DEF2-TZVPP/C", "EnGrad"]

    def test_ccsd_gold_is_canonical_rather_than_dlpno(self):
        """``'CCSD(T)'`` is passed through as an ORCA keyword, unlike
        ``calc_type='CCSD'`` which builds the DLPNO approximation."""
        calc = orca_calc_preset(orca_path="/nonexistent/orca",
                                **orca_preset_ccsd_gold)

        assert calc.parameters["orcasimpleinput"].split() == [
            "CCSD(T)", "DEF2-TZVPP", "EnGrad"]
        assert "DLPNO" not in calc.parameters["orcasimpleinput"]

    def test_a_keyword_after_the_splat_overrides_the_preset(self):
        calc = orca_calc_preset(orca_path="/nonexistent/orca",
                                **orca_preset_dft_cheap,
                                n_procs=8)

        assert "%pal nprocs 8 end" in calc.parameters["orcablocks"]
