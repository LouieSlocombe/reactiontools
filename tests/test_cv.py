"""Tests for the proton-transfer collective variables and the scripts for them.

The builders are pure string construction, so these run without PLUMED, without
an MD engine and without any data files: a three-atom geometry is enough to
size a switching function.
"""

import re
import sys

import numpy as np
import pytest
from ase import Atoms

from reactiontools import (
    PLUMED_ASE_UNITS,
    as_positions,
    plumed_angle_radians,
    plumed_bias_and_fes,
    plumed_input_1pt,
    plumed_input_2pt_1d,
    plumed_input_2pt_2d,
    plumed_input_neb_path,
    plumed_input_steered,
    plumed_input_steered_pt,
    plumed_one_based,
    plumed_temperature_pair,
    plumed_units_header,
    switching_value,
)

DONOR, HYDROGEN, ACCEPTOR = 0, 1, 2

#: Donor, hydrogen, acceptor in a line, in angstrom. The proton sits 1.02 A
#: from the donor and 1.69 A from the acceptor, so the CV starts positive.
#: These are the coordinates the golden script below was captured with.
PT_GEOMETRY = np.array([[0.00, 0.00, 0.00], [1.02, 0.00, 0.00], [2.71, 0.00, 0.00]])

#: Two donor/hydrogen/acceptor triples, with spare atoms around them.
TWO_TRANSFER_GEOMETRY = np.array(
    [
        [0.00, 0.00, 0.00],
        [1.02, 0.00, 0.00],
        [2.71, 0.00, 0.00],
        [0.13, 1.47, 0.00],
        [1.19, 2.05, 0.31],
        [2.83, 1.63, 0.22],
        [-1.21, 0.44, 0.77],
        [3.94, 0.62, -0.55],
        [0.61, -1.38, 1.02],
        [3.31, -1.11, 0.84],
    ]
)

TEMPERATURE = 300.0

#: Every builder, with arguments that exercise it. Used to check the whole
#: family at once for the mistakes that are easy to make in an f-string.
ALL_BUILDERS = [
    ("1pt", plumed_input_1pt, (PT_GEOMETRY, [0, 1, 2], TEMPERATURE)),
    (
        "2pt_1d",
        plumed_input_2pt_1d,
        (TWO_TRANSFER_GEOMETRY, [0, 1, 2], [3, 4, 5], TEMPERATURE),
    ),
    (
        "2pt_2d",
        plumed_input_2pt_2d,
        (TWO_TRANSFER_GEOMETRY, [0, 1, 2], [3, 4, 5], TEMPERATURE),
    ),
    ("neb_path", plumed_input_neb_path, (TEMPERATURE,)),
]

#: The subset that biases with metadynamics, so returns a bias line and a
#: reconstruction command. The steered builders return a step count instead.
_LABEL = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*\S")


def labels_of(script):
    """Every action label defined in a script, in order."""
    return [
        _LABEL.match(line).group(1)
        for line in script.splitlines()
        if not line.lstrip().startswith("#") and _LABEL.match(line)
    ]


class TestEveryBuilder:
    """Checks that should hold for all of them, whatever they bias."""

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_no_placeholder_survives_into_the_script(self, name, builder, args):
        # An f-string that lost a brace, or a value that was never computed,
        # produces a script PLUMED rejects with an unhelpful parse error.
        script, _ = builder(*args)

        assert "{" not in script and "}" not in script
        assert "None" not in script

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_labels_are_unique(self, name, builder, args):
        # PLUMED refers to an action by its label; two actions sharing one is
        # an error it only reports at run time.
        script, _ = builder(*args)
        labels = labels_of(script)

        duplicates = {label for label in labels if labels.count(label) > 1}
        assert not duplicates

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_every_arg_refers_to_something_defined_before_it(self, name, builder, args):
        script, _ = builder(*args)

        defined = set()
        for line in script.splitlines():
            if line.lstrip().startswith("#"):
                continue
            for arg in re.findall(r"ARG=([\w.,]+)", line):
                for term in arg.split(","):
                    # path.sss, metad.bias and dist.z are components of the
                    # action named before the dot.
                    assert term.split(".")[0] in defined, (
                        f"{name}: ARG={term} used before it is defined"
                    )
            match = _LABEL.match(line)
            if match:
                defined.add(match.group(1))

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_indices_are_one_based(self, name, builder, args):
        # Every builder is given a 0-based index 0, which PLUMED must never
        # see: it counts atoms from one, and index 0 is a parse error.
        script, _ = builder(*args)

        for atoms in re.findall(r"(?:ATOMS|GROUPA|GROUPB)=([\d,]+)", script):
            assert "0" not in atoms.split(","), f"{name}: 0-based index leaked"

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_the_bias_and_the_print_agree_on_the_temperature(self, name, builder, args):
        script, _ = builder(*args)

        assert f"TEMP={TEMPERATURE}" in script

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_opes_switches_both_the_bias_and_the_command(self, name, builder, args):
        # The bug this catches: builders that hand-rolled their own METAD line
        # accepted f_opes and then silently ignored it. Everything goes through
        # plumed_bias_and_fes now, so neither half can be switched alone.
        script, command = builder(*args, f_opes=True)

        assert "OPES_METAD" in script
        assert "BARRIER=" in script
        assert "METAD ARG" not in script.replace("OPES_METAD ARG", "")
        assert "FES_from_State.py" in command
        assert command.startswith(sys.executable)

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_well_tempered_metad_is_the_default(self, name, builder, args):
        script, command = builder(*args)

        assert "METAD ARG" in script and "OPES_METAD" not in script
        assert "BIASFACTOR=" in script
        assert command.startswith("plumed sum_hills")

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_the_command_carries_kt_for_this_temperature(self, name, builder, args):
        _, command = builder(*args)

        # 2.4943 kJ/mol at 300 K; the reconstruction reweights by it.
        assert "--kt 2.49434" in command

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_ase_units_change_the_header_and_the_thermal_energy(
        self, name, builder, args
    ):
        script, command = builder(*args, units="ase")

        assert script.lstrip().startswith(PLUMED_ASE_UNITS)
        # kBT at 300 K is 0.025852 eV, not 2.4943 kJ/mol.
        assert "--kt 0.025852" in command

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_plumed_units_write_no_units_line(self, name, builder, args):
        script, _ = builder(*args)

        assert "UNITS" not in script

    @pytest.mark.parametrize(
        "name, builder, args", ALL_BUILDERS, ids=[case[0] for case in ALL_BUILDERS]
    )
    def test_an_unknown_unit_system_is_rejected(self, name, builder, args):
        with pytest.raises(ValueError, match="Unknown units"):
            builder(*args, units="hartree")


class TestGeometrySizing:
    def test_r0_comes_from_the_shorter_bond(self):
        # 1.02 A donor-H and 1.69 A acceptor-H, so R_0 = 1.02 * 1.1 = 1.122 A,
        # which is 0.11 nm to two decimals.
        script, _ = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], TEMPERATURE)

        assert "R_0=0.11" in script

    def test_the_wall_scales_with_the_donor_acceptor_distance(self):
        # 2.71 A apart, wall at 1.5x that is 4.065 A = 0.41 nm.
        script, _ = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], TEMPERATURE)

        assert "UPPER_WALLS ARG=dist_da AT=0.41" in script

    def test_ase_units_scale_the_geometry_by_ten(self):
        script, _ = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], TEMPERATURE, units="ase")

        assert "R_0=1.12" in script
        assert "UPPER_WALLS ARG=dist_da AT=4.06" in script

    def test_the_angle_wall_is_written_in_radians(self):
        script, _ = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], TEMPERATURE)

        # 130 degrees is 2.27 radians
        assert "LOWER_WALLS ARG=ang_1 AT=2.27" in script


class TestAsPositions:
    def test_a_bare_array_is_taken_as_angstrom(self):
        assert np.allclose(as_positions(PT_GEOMETRY), PT_GEOMETRY)

    def test_ase_atoms_are_accepted(self):
        atoms = Atoms("OHO", positions=PT_GEOMETRY)

        assert np.allclose(as_positions(atoms), PT_GEOMETRY)

    def test_an_ase_atoms_and_an_array_give_the_same_script(self):
        atoms = Atoms("OHO", positions=PT_GEOMETRY)

        from_atoms, _ = plumed_input_1pt(atoms, [0, 1, 2], TEMPERATURE)
        from_array, _ = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], TEMPERATURE)

        assert from_atoms == from_array

    def test_a_structure_file_is_read(self, pt_pdb, pt_atoms):
        assert np.allclose(as_positions(str(pt_pdb)), pt_atoms.positions, atol=1e-3)

    def test_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match=r"\(n_atoms, 3\)"):
            as_positions(np.zeros((3, 4)))

    def test_openmm_quantities_are_converted_from_nanometres(self):
        # Built without importing OpenMM: as_positions dispatches on the
        # module a positions object came from, so a stand-in with the same
        # shape exercises the same branch.
        openmm = pytest.importorskip("openmm")
        from openmm import unit as openmm_unit

        quantity = openmm_unit.Quantity(
            [openmm.Vec3(*row) for row in PT_GEOMETRY * 0.1], openmm_unit.nanometer
        )

        class Modeller:
            positions = quantity

        assert np.allclose(as_positions(Modeller()), PT_GEOMETRY)


class TestTemperature:
    def test_a_bare_number_is_taken_as_kelvin(self):
        # The old builders required an OpenMM Quantity and crashed on a float.
        script, _ = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], 300.0)

        assert "TEMP=300.0" in script

    def test_an_openmm_quantity_is_still_accepted(self):
        openmm_unit = pytest.importorskip("openmm.unit")

        from_quantity, _ = plumed_input_1pt(
            PT_GEOMETRY, [0, 1, 2], 300.0 * openmm_unit.kelvin
        )
        from_float, _ = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], 300.0)

        assert from_quantity == from_float


class TestSwitchingValue:
    def test_a_bonded_distance_is_close_to_one(self):
        assert switching_value(0.5, 1.0) > 0.9

    def test_a_long_distance_is_close_to_zero(self):
        assert switching_value(5.0, 1.0) < 0.01

    def test_r_equal_to_r0_takes_the_limit_rather_than_dividing_by_zero(self):
        # Both halves vanish there; the limit is nn / mm.
        assert switching_value(1.0, 1.0) == pytest.approx(0.5)

    def test_it_decreases_with_distance(self):
        values = [switching_value(r, 1.0) for r in np.linspace(0.2, 3.0, 20)]

        assert np.all(np.diff(values) < 0)


class TestSteered:
    def test_it_schedules_the_pull(self):
        script, n_steps = plumed_input_steered(
            "pt_cv: DISTANCE ATOMS=1,2",
            1.0,
            -1.0,
            5_000,
            cv_name="pt_cv",
            steps_equil=1_000,
            steps_relax=500,
            stride=50,
        )

        assert n_steps == 6_500
        assert "MOVINGRESTRAINT ARG=pt_cv" in script
        # Hold, pull, hold: four milestones, the last of them at the end value
        assert "STEP0=0 AT0=1.0000" in script
        assert "STEP1=1000 AT1=1.0000" in script
        assert "STEP2=6000 AT2=-1.0000" in script
        assert "STEP3=6500 AT3=-1.0000" in script
        assert (
            "PRINT       ARG=pt_cv,smd.pt_cv_cntr,smd.work STRIDE=50 FILE=COLVAR_SMD"
            in script
        )

    def test_without_holds_it_has_two_milestones(self):
        script, n_steps = plumed_input_steered("cv: DISTANCE ATOMS=1,2", 0.2, 0.5, 100)

        assert n_steps == 100
        assert "STEP0=0 AT0=0.2000" in script
        assert "STEP1=100 AT1=0.5000" in script
        assert "STEP2=" not in script

    def test_pt_reads_the_ends_off_the_geometry(self):
        script, n_steps = plumed_input_steered_pt(
            PT_GEOMETRY, [DONOR, HYDROGEN, ACCEPTOR], 10_000
        )

        assert n_steps == 10_000
        # PLUMED counts atoms from one
        assert f"COORDINATION GROUPA={DONOR + 1} GROUPB={HYDROGEN + 1}" in script
        assert f"COORDINATION GROUPA={ACCEPTOR + 1} GROUPB={HYDROGEN + 1}" in script
        assert "UPPER_WALLS ARG=dist_da" in script

        # The proton starts on the donor, so the CV starts positive and is
        # pulled to the mirror image of wherever it started
        at_values = [float(value) for value in re.findall(r"\bAT\d+=(\S+)", script)]
        assert at_values[0] > 0.0
        assert at_values[-1] == pytest.approx(-at_values[0])

    def test_explicit_ends_override_the_geometry(self):
        script, _ = plumed_input_steered_pt(
            PT_GEOMETRY, [0, 1, 2], 100, cv_start=0.9, cv_stop=-0.4
        )

        assert "AT0=0.9000" in script
        assert "AT1=-0.4000" in script


class TestGoldenScript:
    """The full text of one script, to pin the shared helpers.

    Every builder in the coordination family is assembled from the same
    private pieces, so this is what stops a change to them drifting the
    output of all nine at once.
    """

    def test_one_proton_transfer_at_300_kelvin(self):
        script, command = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], TEMPERATURE)

        assert (
            script
            == """
# Proton transfer
c_d:        COORDINATION GROUPA=1 GROUPB=2 R_0=0.11
c_a:        COORDINATION GROUPA=3 GROUPB=2 R_0=0.11
pt_cv:      COMBINE ARG=c_d,c_a COEFFICIENTS=1,-1 PERIODIC=NO

# Limits
dist_da:    DISTANCE ATOMS=3,1
dist_wall:  UPPER_WALLS ARG=dist_da AT=0.41 KAPPA=500.0
ang_1:      ANGLE ATOMS=3,2,1
ang_wall:   LOWER_WALLS ARG=ang_1 AT=2.27 KAPPA=500.0

# Metadynamics
metad:      METAD ARG=pt_cv PACE=500 HEIGHT=15.0 SIGMA=0.05 BIASFACTOR=20.0 TEMP=300.0 FILE=HILLS GRID_MIN=-1.1 GRID_MAX=1.1 GRID_BIN=200
PRINT       ARG=c_d,c_a,pt_cv,metad.bias STRIDE=500 FILE=COLVAR
        """
        )
        assert command == (
            "plumed sum_hills --hills HILLS --outfile fes.dat "
            "--min -1.1 --max 1.1 --bin 200 --kt 2.49434"
        )


class TestGrids:
    def test_the_builders_with_bounds_pass_them_to_both_halves(self):
        script, command = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], TEMPERATURE)

        assert "GRID_MIN=-1.1 GRID_MAX=1.1 GRID_BIN=200" in script
        assert "--min -1.1 --max 1.1 --bin 200" in command

    def test_a_builder_without_bounds_omits_them_from_both_halves(self):
        # Passing no bounds lets PLUMED size its own grid. The bug this pins:
        # a hand-rolled command kept --bin while dropping --min/--max, so the
        # two halves disagreed about the grid.
        script, command = plumed_input_1pt(
            PT_GEOMETRY, [0, 1, 2], TEMPERATURE, grid_min=None, grid_max=None
        )

        assert "GRID_MIN" not in script
        assert "--min" not in command and "--max" not in command
        assert "--bin 200" in command

    def test_the_two_dimensional_builder_doubles_every_grid_setting(self):
        script, command = plumed_input_2pt_2d(
            TWO_TRANSFER_GEOMETRY, [0, 1, 2], [3, 4, 5], TEMPERATURE
        )

        assert "ARG=cv_diff1,cv_diff2" in script
        assert "SIGMA=0.05,0.05" in script
        assert "GRID_MIN=-1.1,-1.1 GRID_MAX=1.1,1.1 GRID_BIN=200,200" in script
        assert "--min -1.1,-1.1 --max 1.1,1.1 --bin 200,200" in command


class TestWhatEachBuilderBiases:
    def test_one_transfer_biases_the_coordination_difference(self):
        script, _ = plumed_input_1pt(PT_GEOMETRY, [0, 1, 2], TEMPERATURE)

        assert "METAD ARG=pt_cv" in script
        assert "pt_cv:      COMBINE ARG=c_d,c_a COEFFICIENTS=1,-1" in script

    def test_two_transfers_in_one_dimension_average_them(self):
        script, _ = plumed_input_2pt_1d(
            TWO_TRANSFER_GEOMETRY, [0, 1, 2], [3, 4, 5], TEMPERATURE
        )

        assert "METAD ARG=pt_cv" in script
        assert "COMBINE ARG=cv_diff1,cv_diff2 COEFFICIENTS=0.5,0.5" in script

    def test_two_transfers_in_two_dimensions_keep_them_apart(self):
        script, _ = plumed_input_2pt_2d(
            TWO_TRANSFER_GEOMETRY, [0, 1, 2], [3, 4, 5], TEMPERATURE
        )

        assert "METAD ARG=cv_diff1,cv_diff2" in script
        assert "COEFFICIENTS=0.5,0.5" not in script

    def test_the_path_builder_biases_progress_and_walls_the_deviation(self):
        script, _ = plumed_input_neb_path(TEMPERATURE)

        assert "METAD ARG=path.sss" in script
        assert "UPPER_WALLS ARG=path.zzz" in script
        assert "PATHMSD REFERENCE=neb_path.pdb" in script
        assert "FIT_TO_TEMPLATE REFERENCE=index_atoms.pdb" in script


class TestTheBuildingBlocks:
    """The plumbing a study builds its own collective variable on."""

    def test_indices_become_one_based_in_the_order_given(self):
        # Order matters: a CV reads its indices positionally, as donor,
        # hydrogen, acceptor. Sorting them would silently rebuild the CV.
        assert plumed_one_based([7, 2, 5]) == [8, 3, 6]

    def test_the_units_header_is_only_written_for_ase(self):
        assert plumed_units_header("plumed") == ""
        assert plumed_units_header("ase").startswith(PLUMED_ASE_UNITS)

    def test_the_temperature_pair_carries_the_energy_unit(self):
        kelvin, kt = plumed_temperature_pair(TEMPERATURE, "plumed")
        assert kelvin == TEMPERATURE
        assert kt == pytest.approx(2.49434, abs=1e-5)

        _, kt_ase = plumed_temperature_pair(TEMPERATURE, "ase")
        assert kt_ase == pytest.approx(0.025852, abs=1e-6)

    def test_angles_are_converted_to_radians(self):
        assert plumed_angle_radians(180.0) == pytest.approx(3.14, abs=1e-2)

    def test_the_bias_and_the_command_agree_about_the_grid(self):
        line, command = plumed_bias_and_fes(
            False,
            "z",
            pace=500,
            height=15.0,
            sigma=0.05,
            bias=20.0,
            temperature=TEMPERATURE,
            kt=2.49434,
            grid_bin=200,
            grid_min=-0.3,
            grid_max=0.3,
        )

        assert "METAD ARG=z" in line
        assert "GRID_MIN=-0.3 GRID_MAX=0.3 GRID_BIN=200" in line
        assert "--min -0.3 --max 0.3 --bin 200" in command

    def test_opes_switches_both_halves(self):
        line, command = plumed_bias_and_fes(
            True,
            "z",
            pace=500,
            height=15.0,
            sigma=0.05,
            bias=20.0,
            temperature=TEMPERATURE,
            kt=2.49434,
            grid_bin=200,
        )

        assert "OPES_METAD ARG=z" in line and "BARRIER=15.0" in line
        assert "FES_from_State.py" in command

    def test_a_study_can_build_a_script_from_them_alone(self):
        # What a downstream collective variable looks like: its own CV lines,
        # this module's plumbing for everything around them.
        donor, hydrogen, acceptor = plumed_one_based([4, 5, 6])
        kelvin, kt = plumed_temperature_pair(TEMPERATURE, "plumed")
        line, command = plumed_bias_and_fes(
            False, "z", 500, 15.0, 0.05, 20.0, kelvin, kt, 200
        )
        script = (
            f"{plumed_units_header('plumed')}"
            f"d1: DISTANCE ATOMS={donor},{hydrogen}\n"
            f"d2: DISTANCE ATOMS={acceptor},{hydrogen}\n"
            f"z: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO\n"
            f"{line}\n"
        )

        assert "{" not in script and "}" not in script
        assert set(labels_of(script)) == {"d1", "d2", "z", "metad"}
        assert command.startswith("plumed sum_hills")
