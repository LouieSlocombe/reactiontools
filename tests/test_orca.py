"""Tests for reactiontools.tools_orca.

Nearly everything here is offline: the keyword-assembly tests point ORCA at a
dummy binary that passes :func:`_resolve_orca`'s checks, and the parser and
extrapolation tests run against captured output. Three integration tests shell
out to the real binary and are skipped unless ``ORCA_PATH`` points at it.

The reference energies are for ``tests/data/fad.xyz`` at the default
``r2SCAN-3c`` level, and are tied to both that geometry and the ORCA version
that produced them.
"""

import os
from math import exp, sqrt
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.orca import OrcaTemplate
from ase.io import read, write

from reactiontools import (
    CHEAP_METHODS,
    NATIVE_XTB_METHODS,
    GoldStandard,
    orca_calc_preset,
    orca_calculate_goat,
    orca_calculator,
    orca_cheap_calculator,
    orca_gold_standard,
    orca_optimise_atoms,
    orca_preset_ccsd_gold,
    orca_preset_dft_cheap,
    orca_preset_dft_gold,
    orca_preset_mp2_gold,
    orca_preset_xtb,
    reaction_energy,
    sella_ts_search,
    tools_orca,
)
from reactiontools.tools_orca import (
    EH_TO_KCAL,
    _basis_name,
    _cbs_params,
    _correlation_energy,
    _extrapolate_corr,
    _extrapolate_scf,
    _find_xtb,
    _geometry_keywords,
    _is_native_xtb,
    _parse_orca,
    _QuietOrcaTemplate,
    _resolve_orca,
    _terminated_normally,
)

# Skip rather than fail: ORCA is licensed separately and installed by hand, so
# a machine without it is the normal case, not a broken one.
orca_required = pytest.mark.skipif(
    os.environ.get("ORCA_PATH") is None,
    reason="needs ORCA; set ORCA_PATH to the executable",
)

DATA = Path(__file__).parent / "data"


@pytest.fixture
def fad() -> Atoms:
    """Formic acid dimer, the reference geometry for the ORCA tests."""
    return read(DATA / "fad.xyz")


@pytest.fixture
def fake_orca(tmp_path: Path) -> Path:
    """Provide a stand-in binary that passes _resolve_orca's ELF check.

    Returns
    -------
    pathlib.Path
        Path to an executable file with an ELF magic number and no content.
    """
    binary = tmp_path / "orca"
    binary.write_bytes(b"\x7fELF" + b"\0" * 64)
    binary.chmod(0o755)
    return binary


@pytest.fixture
def external_xtb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pretend an external xtb driver is installed, via $XTBEXE.

    Returns
    -------
    pathlib.Path
        Path to the stand-in driver.
    """
    driver = tmp_path / "xtb_driver"
    driver.write_bytes(b"\x7fELF")
    driver.chmod(0o755)
    monkeypatch.setenv("XTBEXE", str(driver))
    return driver


@pytest.fixture
def no_external_xtb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend no external xtb driver exists anywhere on this machine."""
    monkeypatch.delenv("XTBEXE", raising=False)
    monkeypatch.setattr("reactiontools.tools_orca.shutil.which", lambda name: None)


def keywords(fake_orca: Path, **kwargs: Any) -> str:
    """Return the ORCA ``!`` line that orca_cheap_calculator would write.

    Parameters
    ----------
    fake_orca : pathlib.Path
        The dummy binary from the :func:`fake_orca` fixture.
    **kwargs
        Passed straight to :func:`orca_cheap_calculator`.

    Returns
    -------
    str
        Contents of the ``orcasimpleinput`` parameter.
    """
    calc = orca_cheap_calculator(orca_path=fake_orca, **kwargs)
    return calc.parameters["orcasimpleinput"]


# --- integration tests, needing the real binary ------------------------------


@pytest.mark.integration
@orca_required
def test_orca_calc_preset(fad: Atoms) -> None:
    fad.calc = orca_calc_preset()
    energy = fad.get_potential_energy()

    assert np.allclose(energy, -10325.045291755621)


@pytest.mark.integration
@orca_required
def test_orca_optimise_atoms(fad: Atoms) -> None:
    opt_atoms = orca_optimise_atoms(fad)
    opt_atoms.calc = orca_calc_preset()
    energy = opt_atoms.get_potential_energy()

    # Relaxing can only lower the energy from the single point above.
    assert np.allclose(energy, -10326.977956847948)


@pytest.mark.integration
@orca_required
def test_orca_calculate_goat(fad: Atoms) -> None:
    conformers, df = orca_calculate_goat(fad)

    assert len(conformers) == len(df)
    assert list(df.columns) == ["Conformer", "Energy_kcal_mol", "Percent_total"]


class TestResolveOrca:
    """Binary discovery, shared by every calculator in the module."""

    def test_orca_path_env_may_be_the_binary_itself(
        self, fake_orca: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ASE_ORCA_COMMAND", raising=False)
        monkeypatch.delenv("ORCA_COMMAND", raising=False)
        monkeypatch.delenv("ORCA_DIR", raising=False)
        monkeypatch.setenv("ORCA_PATH", str(fake_orca))

        assert _resolve_orca(None) == str(fake_orca)

    def test_orca_path_env_may_be_the_install_directory(
        self, fake_orca: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ASE_ORCA_COMMAND", raising=False)
        monkeypatch.delenv("ORCA_COMMAND", raising=False)
        monkeypatch.delenv("ORCA_DIR", raising=False)
        monkeypatch.setenv("ORCA_PATH", str(fake_orca.parent))

        assert _resolve_orca(None) == str(fake_orca)

    def test_screen_reader_is_rejected(self) -> None:
        screen_reader = "/usr/bin/orca"
        if not Path(screen_reader).is_file():
            pytest.skip("no /usr/bin/orca on this machine")
        with pytest.raises(RuntimeError, match="screen reader"):
            _resolve_orca(screen_reader)

    def test_missing_binary_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _resolve_orca("/nonexistent/path/to/orca")


class TestOrcaCalcPreset:
    """The input assembly is pure string work, so it runs without ORCA."""

    def test_builds_the_simple_input_line(self, fake_orca: Path) -> None:
        calc = orca_calc_preset(
            orca_path=fake_orca,
            xc="PBE0",
            basis_set="def2-TZVP",
            f_disp=True,
            f_solv="TOLUENE",
            n_procs=4,
        )

        simple = calc.parameters["orcasimpleinput"]
        blocks = calc.parameters["orcablocks"]

        assert simple.split() == ["PBE0", "D4", "def2-TZVP", "EnGrad"]
        assert "%pal nprocs 4 end" in blocks
        assert 'SMDSOLVENT "TOLUENE"' in blocks

    def test_defaults_solvation_and_dispersion_keywords(self, fake_orca: Path) -> None:
        """``True`` picks the default; a string is passed through as given."""
        default = orca_calc_preset(orca_path=fake_orca, f_solv=True, f_disp=True)

        assert 'SMDSOLVENT "WATER"' in default.parameters["orcablocks"]
        assert "D4" in default.parameters["orcasimpleinput"]

        named = orca_calc_preset(orca_path=fake_orca, f_disp="D3BJ")

        assert "D3BJ" in named.parameters["orcasimpleinput"]

    def test_omits_solvation_and_dispersion_by_default(self, fake_orca: Path) -> None:
        calc = orca_calc_preset(orca_path=fake_orca)

        assert "SMD" not in calc.parameters["orcablocks"]
        assert "D4" not in calc.parameters["orcasimpleinput"]

    def test_uses_an_unrestricted_reference_when_open_shell(
        self, fake_orca: Path
    ) -> None:
        calc = orca_calc_preset(orca_path=fake_orca, multiplicity=3)

        assert calc.parameters["orcasimpleinput"].startswith("UKS ")

    def test_builds_a_qmmm_region(self, fake_orca: Path) -> None:
        calc = orca_calc_preset(
            orca_path=fake_orca, calc_type="QM/XTB2", atom_list="0:5"
        )

        assert calc.parameters["orcasimpleinput"].startswith("QM/XTB2 ")
        assert "%QMMM QMATOMS {0:5} END END" in calc.parameters["orcablocks"]

    def test_rejects_a_bogus_binary_at_construction(self) -> None:
        """Routing through the resolver moves the failure up front."""
        with pytest.raises(FileNotFoundError):
            orca_calc_preset(orca_path="/nonexistent/orca")


class TestPresets:
    """The ``orca_preset_*`` dictionaries, splatted into orca_calc_preset.

    Like the tests above these are pure string assembly, so they run without
    ORCA. They pin the level of theory each preset names, which is the whole
    point of having them: a preset that quietly changed functional would
    change every result taken with it.
    """

    def test_dft_cheap_is_blyp_in_the_gas_phase(self, fake_orca: Path) -> None:
        calc = orca_calc_preset(orca_path=fake_orca, **orca_preset_dft_cheap)

        assert calc.parameters["orcasimpleinput"].split() == [
            "BLYP",
            "6-31+G(d,p)",
            "EnGrad",
        ]
        assert "SMD" not in calc.parameters["orcablocks"]

    def test_dft_gold_is_b3lyp_with_d4_in_water(self, fake_orca: Path) -> None:
        calc = orca_calc_preset(orca_path=fake_orca, **orca_preset_dft_gold)

        assert calc.parameters["orcasimpleinput"].split() == [
            "B3LYP",
            "D4",
            "DEF2-SVP",
            "EnGrad",
        ]
        assert 'SMDSOLVENT "WATER"' in calc.parameters["orcablocks"]

    def test_xtb_names_the_method_and_takes_no_basis(self, fake_orca: Path) -> None:
        calc = orca_calc_preset(orca_path=fake_orca, **orca_preset_xtb)

        assert calc.parameters["orcasimpleinput"].split() == ["XTB2", "EnGrad"]

    def test_mp2_gold_uses_the_dlpno_approximation_and_an_aux_basis(
        self, fake_orca: Path
    ) -> None:
        calc = orca_calc_preset(orca_path=fake_orca, **orca_preset_mp2_gold)

        assert calc.parameters["orcasimpleinput"].split() == [
            "DLPNO-MP2",
            "DEF2-TZVPP",
            "DEF2-TZVPP/C",
            "EnGrad",
        ]

    def test_ccsd_gold_is_canonical_rather_than_dlpno(self, fake_orca: Path) -> None:
        """``'CCSD(T)'`` is passed through as an ORCA keyword, unlike
        ``calc_type='CCSD'`` which builds the DLPNO approximation."""
        calc = orca_calc_preset(orca_path=fake_orca, **orca_preset_ccsd_gold)

        assert calc.parameters["orcasimpleinput"].split() == [
            "CCSD(T)",
            "DEF2-TZVPP",
            "EnGrad",
        ]
        assert "DLPNO" not in calc.parameters["orcasimpleinput"]

    def test_a_keyword_after_the_splat_overrides_the_preset(
        self, fake_orca: Path
    ) -> None:
        calc = orca_calc_preset(orca_path=fake_orca, **orca_preset_dft_cheap, n_procs=8)

        assert "%pal nprocs 8 end" in calc.parameters["orcablocks"]


class TestOrcaCheapCalculator:
    """Keyword assembly for the xTB / "3c" screening tier."""

    def test_every_method_maps_to_its_orca_keyword(
        self, fake_orca: Path, external_xtb: Path
    ) -> None:
        for alias, (keyword, _) in CHEAP_METHODS.items():
            assert keyword in keywords(fake_orca, method=alias, native=False).split()

    def test_unknown_method_is_rejected(self, fake_orca: Path) -> None:
        with pytest.raises(ValueError, match="pick one of"):
            keywords(fake_orca, method="ccsd(t)")

    def test_method_alias_is_case_insensitive(self, fake_orca: Path) -> None:
        assert "r2SCAN-3c" in keywords(fake_orca, method="R2SCAN-3C ")
        assert "NATIVE-XTB2" in keywords(fake_orca, method=" GFN2-XTB ")

    # --- native vs external xTB ----------------------------------------------

    def test_gfn_levels_default_to_the_native_implementation(
        self, fake_orca: Path
    ) -> None:
        for alias, (plain, spin_polarised) in NATIVE_XTB_METHODS.items():
            assert plain in keywords(fake_orca, method=alias).split()
            assert plain in keywords(fake_orca, method=alias, native=True).split()
            assert (
                spin_polarised
                in keywords(fake_orca, method=alias, spin_polarised=True).split()
            )

    def test_external_route_writes_the_plain_keyword(
        self, fake_orca: Path, external_xtb: Path
    ) -> None:
        line = keywords(fake_orca, method="gfn2-xtb", native=False).split()
        assert "XTB2" in line
        assert "NATIVE-XTB2" not in line

    def test_gfn_ff_is_external_only(self, fake_orca: Path, external_xtb: Path) -> None:
        assert "XTBFF" in keywords(fake_orca, method="gfn-ff").split()
        with pytest.raises(ValueError, match="no native ORCA implementation"):
            keywords(fake_orca, method="gfn-ff", native=True)

    def test_native_is_ignored_for_the_composites(
        self, fake_orca: Path, no_external_xtb: None
    ) -> None:
        for native in (True, False, "auto"):
            assert (
                "r2SCAN-3c"
                in keywords(fake_orca, method="r2scan-3c", native=native).split()
            )

    def test_bad_native_value_is_rejected(self, fake_orca: Path) -> None:
        with pytest.raises(ValueError, match="native must be True, False or 'auto'"):
            keywords(fake_orca, native="yes")

    def test_spin_polarisation_needs_the_native_route(
        self, fake_orca: Path, external_xtb: Path
    ) -> None:
        with pytest.raises(ValueError, match="needs ORCA's native xTB"):
            keywords(fake_orca, method="gfn2-xtb", native=False, spin_polarised=True)
        with pytest.raises(ValueError, match="needs ORCA's native xTB"):
            keywords(fake_orca, method="gfn-ff", spin_polarised=True)
        with pytest.raises(ValueError, match="needs ORCA's native xTB"):
            keywords(fake_orca, method="r2scan-3c", spin_polarised=True)

    def test_missing_external_driver_is_reported_up_front(
        self, fake_orca: Path, no_external_xtb: None
    ) -> None:
        with pytest.raises(FileNotFoundError, match="native=True"):
            keywords(fake_orca, method="gfn2-xtb", native=False)
        with pytest.raises(FileNotFoundError, match="no native implementation"):
            keywords(fake_orca, method="gfn-ff")
        # The native route never touches the external interface, so it still builds.
        assert "NATIVE-XTB2" in keywords(fake_orca, method="gfn2-xtb").split()

    def test_xtb_block_belongs_to_the_external_interface(
        self, fake_orca: Path, external_xtb: Path
    ) -> None:
        with pytest.raises(ValueError, match="%xtb block"):
            keywords(fake_orca, method="gfn2-xtb", extra_blocks="%xtb ETemp 300 end")
        calc = orca_cheap_calculator(
            "gfn2-xtb",
            orca_path=fake_orca,
            native=False,
            extra_blocks="%xtb ETemp 300 end",
        )
        assert "%xtb ETemp 300 end" in calc.parameters["orcablocks"]

    # --- keyword line and blocks ---------------------------------------------

    def test_engrad_is_requested_by_default(self, fake_orca: Path) -> None:
        assert "EnGrad" in keywords(fake_orca).split()
        assert "EnGrad" not in keywords(fake_orca, forces=False).split()

    def test_solvent_model_defaults_per_method(
        self, fake_orca: Path, external_xtb: Path
    ) -> None:
        assert "ALPB(water)" in keywords(fake_orca, method="gfn2-xtb", solvent="water")
        assert "ALPB(water)" in keywords(fake_orca, method="gfn-ff", solvent="water")
        assert "CPCM(water)" in keywords(fake_orca, method="r2scan-3c", solvent="water")
        assert "CPCM(water)" in keywords(fake_orca, method="hf-3c", solvent="water")

    def test_solvent_model_can_be_overridden(self, fake_orca: Path) -> None:
        assert "SMD(toluene)" in keywords(
            fake_orca, method="b97-3c", solvent="toluene", solvent_model="SMD"
        )

    def test_no_solvent_keyword_without_a_solvent(self, fake_orca: Path) -> None:
        for token in ("CPCM", "ALPB", "SMD"):
            assert token not in keywords(fake_orca)

    def test_charge_multiplicity_and_blocks(self, fake_orca: Path) -> None:
        calc = orca_cheap_calculator(
            "hf-3c",
            orca_path=fake_orca,
            charge=-1,
            multiplicity=2,
            n_procs=8,
            maxcore=2000,
            scf_convergence="TightSCF",
            extra_keywords="SlowConv",
            extra_blocks="%scf MaxIter 300 end",
        )
        assert calc.parameters["charge"] == -1
        assert calc.parameters["mult"] == 2
        simple = calc.parameters["orcasimpleinput"]
        assert "TightSCF" in simple and "SlowConv" in simple
        blocks = calc.parameters["orcablocks"]
        assert "%pal nprocs 8 end" in blocks
        assert "%maxcore 2000" in blocks
        assert "%scf MaxIter 300 end" in blocks

    def test_quiet_template_is_opt_out(self, fake_orca: Path) -> None:
        assert isinstance(
            orca_cheap_calculator(orca_path=fake_orca).template, _QuietOrcaTemplate
        )
        plain = orca_cheap_calculator(orca_path=fake_orca, quiet=False)
        assert not isinstance(plain.template, _QuietOrcaTemplate)


class TestQuietTemplate:
    """The stdout-suppressing wrapper around ASE's ORCA template."""

    def test_swallows_the_engrad_print(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def read(self, directory: str | Path) -> dict[str, float]:
            print("ORCA does not by default supply the forces")
            return {"energy": 1.0}

        monkeypatch.setattr(OrcaTemplate, "read_results", read)
        assert _QuietOrcaTemplate().read_results("somewhere") == {"energy": 1.0}
        assert capsys.readouterr().out == ""

    def test_still_propagates_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class Boom(OSError):
            pass

        def read(self, directory: str | Path) -> NoReturn:
            print("noise")
            raise Boom("real failure")

        monkeypatch.setattr(OrcaTemplate, "read_results", read)
        with pytest.raises(Boom, match="real failure"):
            _QuietOrcaTemplate().read_results("somewhere")
        assert capsys.readouterr().out == ""


class TestFindXtb:
    """Discovery of the external xtb driver, mirroring ORCA's own search."""

    def test_searches_where_orca_does(
        self,
        fake_orca: Path,
        tmp_path: Path,
        no_external_xtb: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert _find_xtb(fake_orca) is None

        sibling = fake_orca.parent / "otool_xtb"
        sibling.write_bytes(b"\x7fELF")
        assert _find_xtb(fake_orca) == str(sibling)

        # $XTBEXE outranks whatever sits next to the ORCA binary.
        elsewhere = tmp_path / "opt" / "xtb"
        elsewhere.parent.mkdir()
        elsewhere.write_bytes(b"\x7fELF")
        monkeypatch.setenv("XTBEXE", str(elsewhere))
        assert _find_xtb(fake_orca) == str(elsewhere)


def geometry_line(method: str, **kwargs: Any) -> list[str]:
    """Return the geometry-stage ``!`` line as a list of keywords.

    Parameters
    ----------
    method : str
        Level of theory for the stage.
    **kwargs
        Overrides for :func:`_geometry_keywords`; the defaults optimise a
        minimum and run frequencies with TightSCF.

    Returns
    -------
    list of str
        The keywords that would be written.
    """
    settings = dict(
        opt_basis=None,
        optimise=True,
        frequencies=True,
        transition_state=False,
        common="TightSCF",
    )
    settings.update(kwargs)
    return _geometry_keywords(opt_method=method, **settings).split()


class TestGeometryKeywords:
    """The gold standard's geometry / thermochemistry stage."""

    def test_takes_native_xtb_and_its_numerical_hessian(self) -> None:
        line = geometry_line("gfn2-xtb")
        assert "NATIVE-XTB2" in line
        # Native xTB has no analytic Hessian, and ORCA aborts rather than fall back.
        assert "NumFreq" in line and "Freq" not in line
        # An xTB Hamiltonian brings its own basis, and refuses to run with RI.
        assert (
            "def2-TZVP" not in line and "RIJCOSX" not in line and "def2/J" not in line
        )

    def test_keeps_the_analytic_hessian_for_everything_else(self) -> None:
        composite = geometry_line("r2SCAN-3c")
        assert composite[:1] == ["r2SCAN-3c"]
        assert "Freq" in composite and "NumFreq" not in composite
        assert "def2-TZVP" not in composite

        functional = geometry_line("PBE0", frequencies=False, transition_state=True)
        assert functional[:4] == ["PBE0", "def2-TZVP", "RIJCOSX", "def2/J"]
        assert "OptTS" in functional and "TightOpt" not in functional

    def test_passes_orca_keywords_through(self) -> None:
        assert "XTB2" in geometry_line("XTB2")
        assert "TightOpt" in geometry_line("XTB2")
        assert geometry_line("wB97X-3c", optimise=False, frequencies=False) == [
            "wB97X-3c",
            "TightSCF",
        ]

    def test_is_native_xtb(self) -> None:
        assert _is_native_xtb("NATIVE-XTB2")
        assert _is_native_xtb("native-spxtb1 TightSCF")
        assert not _is_native_xtb("XTB2")
        assert not _is_native_xtb("r2SCAN-3c")


def mechanism_input(fake_orca: Path, **kwargs: Any) -> tuple[str, str]:
    """Return the simple-input and block strings from orca_calculator."""
    calc = orca_calculator(orca_path=fake_orca, **kwargs)
    return calc.parameters["orcasimpleinput"], calc.parameters["orcablocks"]


class TestOrcaCalculator:
    """Offline coverage of every mechanism-workflow configuration branch."""

    def test_default_is_an_ase_gradient_at_the_documented_level(
        self, fake_orca: Path
    ) -> None:
        simple, blocks = mechanism_input(fake_orca)

        assert simple.split()[:6] == [
            "wB97M-V",
            "def2-TZVPD",
            "RIJCOSX",
            "def2/J",
            "NoUseSym",
            "DEFGRID3",
        ]
        assert "EnGrad" in simple
        assert "NONBO NONPA" in simple
        assert "%scf MaxIter 300 end" in blocks

    @pytest.mark.parametrize(
        ("strategy", "included", "excluded"),
        [
            ("default", [], ["SlowConv", "NOSOSCF"]),
            ("omol", ["DIIS", "NOSOSCF", "NormalConv"], ["SlowConv"]),
            ("slow", ["SlowConv"], ["NOSOSCF"]),
        ],
    )
    def test_scf_strategies(
        self, fake_orca: Path, strategy: str, included: list[str], excluded: list[str]
    ) -> None:
        simple, _ = mechanism_input(fake_orca, scf_strategy=strategy)

        assert all(keyword in simple.split() for keyword in included)
        assert all(keyword not in simple.split() for keyword in excluded)

    def test_rejects_unknown_task_and_scf_strategy(self, fake_orca: Path) -> None:
        with pytest.raises(ValueError, match="task must be one of"):
            orca_calculator(task="dance")
        with pytest.raises(ValueError, match="scf_strategy"):
            orca_calculator(orca_path=fake_orca, scf_strategy="reckless")

    def test_restart_solvation_and_population_controls(self, fake_orca: Path) -> None:
        simple, blocks = mechanism_input(
            fake_orca,
            solvent="water",
            solvation_model="smd",
            autostart=False,
            moread="previous.gbw",
            population=True,
            nbo=True,
        )

        assert {"CPCM(water)", "NoAutoStart", "MORead", "ALLPOP"} <= set(simple.split())
        assert "NONBO" not in simple
        assert 'SMDsolvent "water"' in blocks
        assert '%moinp "previous.gbw"' in blocks
        assert "%output" in blocks and "%nbo" in blocks

    def test_open_shell_singlet_needs_atoms_and_breaks_symmetry(
        self, fake_orca: Path
    ) -> None:
        with pytest.raises(ValueError, match="atoms= is required"):
            orca_calculator(
                orca_path=fake_orca,
                multiplicity=1,
                open_shell_singlet=True,
            )

        simple, blocks = mechanism_input(
            fake_orca,
            atoms=Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]]),
            multiplicity=1,
            open_shell_singlet=True,
        )
        assert "UKS" in simple.split()
        assert "%scf rotate" in blocks

    def test_saddle_controls_share_one_geom_block(self, fake_orca: Path) -> None:
        simple, blocks = mechanism_input(
            fake_orca,
            task="optts+freq",
            hybrid_hess_atoms=[0, 2, 5],
            recalc_hess=10,
            ts_mode=1,
            geom_maxiter=80,
            temperature=310,
            freq_increment=0.01,
        )

        assert {"OptTS", "NumFreq"} <= set(simple.split())
        assert "Calc_Hess true" in blocks
        assert "Hybrid_Hess {0 2 5} end" in blocks
        assert "Recalc_Hess 10" in blocks
        assert "TS_Mode {M 1} end" in blocks
        assert "MaxIter 80" in blocks
        assert "%freq Temp 310" in blocks and "Increment 0.01" in blocks

    def test_stored_hessian_takes_precedence_over_calculating_one(
        self, fake_orca: Path
    ) -> None:
        _, blocks = mechanism_input(
            fake_orca,
            task="optts",
            calc_hess=True,
            inhess_file="start.hess",
        )

        assert 'InHessName "start.hess"' in blocks
        assert "Calc_Hess" not in blocks

    def test_neb_and_irc_requirements_are_encoded(self, fake_orca: Path) -> None:
        with pytest.raises(ValueError, match="neb_product"):
            orca_calculator(orca_path=fake_orca, task="neb-ts")

        _, neb_blocks = mechanism_input(
            fake_orca,
            task="neb-ts",
            neb_product="product.xyz",
            neb_ts_guess="guess.xyz",
            neb_images=12,
        )
        _, irc_blocks = mechanism_input(
            fake_orca,
            task="irc",
            irc_maxiter=90,
            irc_hess_file="ts.hess",
        )
        assert 'NEB_End_XYZFile "product.xyz"' in neb_blocks
        assert 'NEB_TS_XYZFile "guess.xyz"' in neb_blocks
        assert "NImages 12" in neb_blocks
        assert "%irc MaxIter 90 Direction both" in irc_blocks
        assert 'Hess_Filename "ts.hess"' in irc_blocks

    def test_scan_coordinate_is_put_in_the_geom_block(self, fake_orca: Path) -> None:
        _, blocks = mechanism_input(
            fake_orca,
            task="scan",
            scan_coord="B 0 1 = 1.8, 1.0, 9",
        )

        assert "Scan B 0 1 = 1.8, 1.0, 9 end" in blocks


@pytest.mark.parametrize(
    ("sella_kwargs", "expected_internal"),
    [({}, False), ({"internal": True}, True)],
)
def test_sella_search_wires_the_calculator_and_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    water: Atoms,
    sella_kwargs: dict[str, bool],
    expected_internal: bool,
) -> None:
    configured = object()
    seen = {}

    def fake_calculator(**kwargs: Any) -> object:
        seen["calculator"] = kwargs
        return configured

    class FakeSella:
        def __init__(self, atoms: Atoms, **kwargs: Any) -> None:
            seen["optimizer"] = (atoms, kwargs)

        def run(self, **kwargs: Any) -> None:
            seen["run"] = kwargs

    monkeypatch.setattr(tools_orca, "orca_calculator", fake_calculator)
    monkeypatch.setattr("sella.Sella", FakeSella)

    result = sella_ts_search(
        water,
        charge=-1,
        multiplicity=2,
        fmax=0.03,
        steps=17,
        trajectory="sella.traj",
        solvent="water",
        **sella_kwargs,
    )

    assert result is water
    assert water.calc is configured
    assert seen["calculator"] == {
        "charge": -1,
        "multiplicity": 2,
        "solvent": "water",
        "task": "engrad",
        "atoms": water,
    }
    assert seen["optimizer"][1] == {
        "order": 1,
        "internal": expected_internal,
        "trajectory": "sella.traj",
    }
    assert seen["run"] == {"fmax": 0.03, "steps": 17}


DLPNO_OUT = """
----------------
TOTAL SCF ENERGY
----------------
Total Energy       :          -76.05793814 Eh           -2069.63426 eV

 E(0)                                       ...    -76.057938140
 E(CORR)(strong-pairs)                      ...     -0.294518123
 E(CORR)(weak-pairs)                        ...     -0.000512004
 E(CORR)(total)                             ...     -0.295030127
 Triples Correction (T)                     ...     -0.008765432

 E(CCSD)                                    ...    -76.352968267
 E(CCSD(T))                                 ...    -76.361733699

FINAL SINGLE POINT ENERGY      -76.361733699
                             ****ORCA TERMINATED NORMALLY****
"""

RIMP2_OUT = """
----------------
TOTAL SCF ENERGY
----------------
Total Energy       :          -76.05793814 Eh           -2069.63426 eV

 RI-MP2 CORRELATION ENERGY:     -0.280144219 Eh
 SCS-MP2 CORRELATION ENERGY:    -0.271000000 Eh

FINAL SINGLE POINT ENERGY      -76.338082359
                             ****ORCA TERMINATED NORMALLY****
"""

FREQ_OUT = """
-----------------------
VIBRATIONAL FREQUENCIES
-----------------------
   0:         0.00 cm**-1
   6:      -523.44 cm**-1 ***imaginary mode***
   7:      1623.11 cm**-1

Electronic energy                ...    -76.05793814 Eh
Zero point energy                ...      0.02154960 Eh      13.52 kcal/mol
Total Enthalpy                   ...    -76.03284214 Eh
G-E(el)                          ...     -0.00190445 Eh      -1.20 kcal/mol
Final Gibbs free energy          ...    -76.05984259 Eh
                             ****ORCA TERMINATED NORMALLY****
"""


class TestOrcaOutputParsing:
    """The gold standard's regex sweep over captured ORCA outputs."""

    def test_parse_dlpno_ccsdt(self) -> None:
        vals = _parse_orca(DLPNO_OUT)
        assert vals["scf"] == pytest.approx(-76.05793814)
        assert vals["cc_corr"] == pytest.approx(-0.295030127)
        assert vals["triples"] == pytest.approx(-0.008765432)
        assert vals["e_ccsdt"] == pytest.approx(-76.361733699)
        assert vals["e_ccsd"] == pytest.approx(-76.352968267)
        # E(CCSD(T)) - E(SCF), and the E(CORR)+(T) route must agree.
        assert _correlation_energy(vals) == pytest.approx(-0.303795559, abs=1e-9)
        assert vals["cc_corr"] + vals["triples"] == pytest.approx(
            -0.303795559, abs=1e-9
        )

    def test_parse_rimp2_ignores_scs(self) -> None:
        vals = _parse_orca(RIMP2_OUT)
        assert vals["mp2_corr"] == pytest.approx(-0.280144219)
        assert _correlation_energy(vals) == pytest.approx(-0.280144219)

    def test_parse_thermochemistry(self) -> None:
        vals = _parse_orca(FREQ_OUT)
        assert vals["zpe"] == pytest.approx(0.02154960)
        assert vals["gibbs_corr"] == pytest.approx(-0.00190445)
        assert vals["enthalpy"] - vals["thermal_eel"] == pytest.approx(
            0.025096, abs=1e-6
        )


class TestCbsExtrapolation:
    """Two-point CBS extrapolation maths and basis-set bookkeeping."""

    def test_extrapolation_recovers_the_limit(self) -> None:
        alpha, beta = _cbs_params("cc", (3, 4))

        e_cbs, amp = -76.0670, 2.5
        lo = e_cbs + amp * exp(-alpha * sqrt(3))
        hi = e_cbs + amp * exp(-alpha * sqrt(4))
        assert _extrapolate_scf(lo, hi, 3, 4, alpha) == pytest.approx(e_cbs, abs=1e-10)

        c_cbs, amp = -0.3100, 0.6
        lo = c_cbs + amp * 3.0**-beta
        hi = c_cbs + amp * 4.0**-beta
        assert _extrapolate_corr(lo, hi, 3, 4, beta) == pytest.approx(c_cbs, abs=1e-10)

    def test_extrapolation_is_bracketed_by_the_two_bases(self) -> None:
        # CBS limits must lie beyond the larger basis, never between the two.
        alpha, beta = _cbs_params("cc", (3, 4))
        assert _extrapolate_scf(-76.0600, -76.0650, 3, 4, alpha) < -76.0650
        assert _extrapolate_corr(-0.2800, -0.2950, 3, 4, beta) < -0.2950

    def test_basis_names(self) -> None:
        assert _basis_name("cc", 3) == "cc-pVTZ"
        assert _basis_name("aug-cc", 4) == "aug-cc-pVQZ"
        assert _basis_name("def2", 3) == "def2-TZVPP"
        with pytest.raises(ValueError):
            _basis_name("cc", 6)
        with pytest.raises(ValueError):
            _cbs_params("cc", (2, 4))


class TestOrcaOutputHelpers:
    def test_normal_termination_requires_a_readable_banner(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing.out"
        incomplete = tmp_path / "incomplete.out"
        complete = tmp_path / "complete.out"
        incomplete.write_text("ORCA is still running")
        complete.write_text(DLPNO_OUT)

        assert _terminated_normally(missing) is False
        assert _terminated_normally(incomplete) is False
        assert _terminated_normally(complete) is True

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ({"final_corr": -0.3}, -0.3),
            ({"cc_corr": -0.29, "triples": -0.01}, -0.3),
            ({"cc_corr": -0.29}, -0.29),
            ({"mp2_corr": -0.28}, -0.28),
            ({"final_sp": -76.3, "scf": -76.0}, -0.3),
        ],
    )
    def test_correlation_energy_accepts_orcas_output_variants(
        self, values: dict[str, float], expected: float
    ) -> None:
        assert _correlation_energy(values) == pytest.approx(expected)

    def test_correlation_energy_rejects_unrecognised_output(self) -> None:
        with pytest.raises(ValueError, match="no correlation energy"):
            _correlation_energy({"scf": -76.0})

    def test_basis_errors_name_the_bad_family_or_cardinal(self) -> None:
        with pytest.raises(ValueError, match="unknown basis family"):
            _basis_name("made-up", 3)
        with pytest.raises(ValueError, match="no def2 basis"):
            _basis_name("def2", 5)


class TestGoldStandardResult:
    def test_derived_energies_and_summary(self) -> None:
        result = GoldStandard(
            atoms=Atoms("H2"),
            charge=-1,
            multiplicity=2,
            e_hf_cbs=-1.0,
            e_corr_cbs=-0.2,
            e_total=-1.2,
            e_mp2_corr_cbs=-0.18,
            delta_cc=-0.02,
            zpe=0.01,
            enthalpy_correction=0.03,
            gibbs_correction=0.02,
            imaginary_frequencies=[-321.0],
            levels={"correlation": "correlation: test/CBS"},
        )

        assert result.energy == pytest.approx(-1.2 * tools_orca.Hartree)
        assert result.enthalpy == pytest.approx(-1.17)
        assert result.gibbs == pytest.approx(-1.18)
        text = result.summary()
        assert "H2  charge=-1 multiplicity=2" in text
        assert "correlation: test/CBS" in text
        assert "CCSD(T)/CBS" in text
        assert "-321.0 cm^-1" in text

    def test_thermal_properties_are_none_without_frequency_corrections(self) -> None:
        result = GoldStandard(atoms=Atoms("H"), e_total=-0.5)

        assert result.enthalpy is None
        assert result.gibbs is None

    def test_reaction_energy_supports_each_energy_level(self) -> None:
        reactant = GoldStandard(
            Atoms("H"),
            e_total=-1.0,
            enthalpy_correction=0.04,
            gibbs_correction=0.02,
        )
        product = GoldStandard(
            Atoms("H"),
            e_total=-0.9,
            enthalpy_correction=0.03,
            gibbs_correction=0.01,
        )

        assert reaction_energy([reactant], [product], "electronic") == pytest.approx(
            0.1 * EH_TO_KCAL
        )
        assert reaction_energy([reactant], [product], "enthalpy") == pytest.approx(
            0.09 * EH_TO_KCAL
        )
        assert reaction_energy([reactant], [product], "gibbs") == pytest.approx(
            0.09 * EH_TO_KCAL
        )

    def test_reaction_energy_rejects_missing_thermochemistry(self) -> None:
        bare = GoldStandard(Atoms("H"), e_total=-1.0)

        with pytest.raises(ValueError, match="run with frequencies=True"):
            reaction_energy([bare], [bare], "gibbs")
        with pytest.raises(KeyError):
            reaction_energy([bare], [bare], "entropy")


def _write_completed_stage(root: Path, name: str, output: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "orca.out").write_text(output)


class TestOrcaGoldStandard:
    """Exercise the compound workflow from reusable captured ORCA outputs."""

    def test_mp2_cbs_plus_coupled_cluster_correction(
        self, fake_orca: Path, water: Atoms, tmp_path: Path
    ) -> None:
        root = tmp_path / "gold"
        _write_completed_stage(root, "mp2_3", RIMP2_OUT)
        _write_completed_stage(root, "mp2_4", RIMP2_OUT)
        _write_completed_stage(root, "cc_3", DLPNO_OUT)

        result = orca_gold_standard(
            water,
            directory=root,
            orca_path=fake_orca,
            optimise=False,
            frequencies=False,
            verbose=False,
        )

        expected_delta = (
            _correlation_energy(_parse_orca(DLPNO_OUT))
            - _parse_orca(RIMP2_OUT)["mp2_corr"]
        )
        assert set(result.components) == {"mp2_3", "mp2_4", "cc_3"}
        assert result.e_mp2_corr_cbs == pytest.approx(-0.280144219)
        assert result.delta_cc == pytest.approx(expected_delta)
        assert result.e_corr_cbs == pytest.approx(-0.280144219 + expected_delta)
        assert result.e_total == pytest.approx(result.e_hf_cbs + result.e_corr_cbs)
        assert "MP2" in result.levels["correlation"]

    def test_direct_coupled_cluster_extrapolation(
        self, fake_orca: Path, water: Atoms, tmp_path: Path
    ) -> None:
        root = tmp_path / "direct-cc"
        _write_completed_stage(root, "cc_3", DLPNO_OUT)
        _write_completed_stage(root, "cc_4", DLPNO_OUT)

        result = orca_gold_standard(
            water,
            directory=root,
            orca_path=fake_orca,
            optimise=False,
            frequencies=False,
            extrapolate_cc=True,
            verbose=False,
        )

        assert set(result.components) == {"cc_3", "cc_4"}
        assert result.e_mp2_corr_cbs is None
        assert result.delta_cc is None
        assert result.e_corr_cbs == pytest.approx(
            _correlation_energy(_parse_orca(DLPNO_OUT))
        )
        assert "DLPNO-CCSD(T)/CBS" in result.levels["correlation"]

    def test_geometry_and_thermochemistry_are_reused(
        self, fake_orca: Path, water: Atoms, tmp_path: Path
    ) -> None:
        root = tmp_path / "with-geometry"
        _write_completed_stage(root, "opt", FREQ_OUT)
        moved = water.copy()
        moved.positions += [0.2, -0.1, 0.3]
        write(root / "opt" / "orca.xyz", moved)
        _write_completed_stage(root, "mp2_3", RIMP2_OUT)
        _write_completed_stage(root, "mp2_4", RIMP2_OUT)
        _write_completed_stage(root, "cc_3", DLPNO_OUT)

        result = orca_gold_standard(
            water,
            directory=root,
            orca_path=fake_orca,
            optimise=True,
            frequencies=True,
            transition_state=True,
            solvent="water",
            verbose=False,
        )

        assert result.atoms.positions == pytest.approx(moved.positions)
        assert result.imaginary_frequencies == [-523.44]
        assert result.zpe == pytest.approx(0.02154960)
        assert result.gibbs_correction == pytest.approx(-0.00190445)
        assert result.enthalpy_correction == pytest.approx(0.025096)
        assert "OptTS" in result.levels["geometry"]
        assert "CPCM(water)" in result.levels["geometry"]

    def test_non_tabulated_cc_cardinal_adds_a_matching_mp2_stage(
        self, fake_orca: Path, water: Atoms, tmp_path: Path
    ) -> None:
        root = tmp_path / "extra-mp2"
        _write_completed_stage(root, "mp2_3", RIMP2_OUT)
        _write_completed_stage(root, "mp2_4", RIMP2_OUT)
        _write_completed_stage(root, "cc_2", DLPNO_OUT)
        _write_completed_stage(root, "mp2_2", RIMP2_OUT)

        result = orca_gold_standard(
            water,
            directory=root,
            orca_path=fake_orca,
            optimise=False,
            frequencies=False,
            cc_cardinal=2,
            verbose=False,
        )

        assert "mp2_2" in result.components
