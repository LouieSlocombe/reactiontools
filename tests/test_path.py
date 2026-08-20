"""Tests for estimating a path collective variable from steered MD.

The trajectory a steered run would produce is faked -- the proton slides from
the donor to the acceptor at a constant rate -- so the frame selection and the
files it writes can be checked without running any dynamics.

The frame-selection functions are NumPy-only; trajectory readers use MDTraj.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactiontools import (
    cv_from_colvar,
    estimate_path_lambda,
    path_from_steered_md,
    select_frames_by_cv,
    select_frames_by_msd,
)

from .conftest import PT_ACCEPTOR, PT_DONOR, PT_HYDROGEN


def write_colvar(
    path: str | Path,
    cv: Sequence[float] | np.ndarray,
    fields: tuple[str, ...] = ("time", "pt_cv", "smd.pt_cv_cntr", "smd.work"),
) -> None:
    """Write a COLVAR file holding *cv*, in the layout a steered run leaves."""
    with open(path, "w") as handle:
        handle.write("#! FIELDS " + " ".join(fields) + "\n")
        for i, value in enumerate(cv):
            handle.write(f"{i * 0.05:.4f} {value:.6f} {value:.6f} {0.1 * i:.6f}\n")


def fake_steered_traj(
    path: str | Path,
    template_pdb: str | Path,
    n_frames: int = 60,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[Any, np.ndarray]:
    """
    Fake the trajectory of a proton being dragged across a hydrogen bond.

    The proton slides from the donor towards the acceptor at a constant rate,
    which makes the frame that belongs to any given point of the pull known in
    advance. Returns the trajectory and the fraction transferred per frame.
    """
    import mdtraj as md

    reference = md.load(str(template_pdb))
    xyz = np.repeat(reference.xyz, n_frames, axis=0)

    fraction = np.linspace(0.0, 1.0, n_frames)
    start = reference.xyz[0, PT_HYDROGEN]
    finish = reference.xyz[0, PT_ACCEPTOR] + (start - reference.xyz[0, PT_DONOR])
    xyz[:, PT_HYDROGEN] = start + (finish - start) * fraction[:, None]

    if noise:
        xyz += np.random.default_rng(seed).normal(scale=noise, size=xyz.shape)

    traj = md.Trajectory(xyz, reference.topology)
    traj.save_pdb(str(path))
    return traj, fraction


class TestSelectFramesByCv:
    def test_it_spans_the_range_evenly(self) -> None:
        cv = np.linspace(-1.0, 1.0, 101)

        picks = select_frames_by_cv(cv, 11)

        assert list(picks) == list(range(0, 101, 10))

    def test_it_honours_explicit_limits(self) -> None:
        cv = np.linspace(-1.0, 1.0, 101)

        picks = select_frames_by_cv(cv, 3, cv_start=-0.5, cv_stop=0.5)

        assert [round(cv[i], 2) for i in picks] == [-0.5, 0.0, 0.5]

    def test_it_moves_forwards_through_a_noisy_pull(self) -> None:
        rng = np.random.default_rng(1)
        cv = np.linspace(-1.0, 1.0, 200) + rng.normal(scale=0.1, size=200)

        picks = select_frames_by_cv(cv, 15)

        assert np.all(np.diff(picks) > 0), "frames must be ordered along the path"
        assert len(picks) == 15

    def test_it_needs_enough_frames(self) -> None:
        with pytest.raises(ValueError, match="Cannot pick"):
            select_frames_by_cv(np.linspace(0.0, 1.0, 5), 10)


class TestSelectFramesByMsd:
    def test_it_spaces_frames_by_displacement(self) -> None:
        # One atom that accelerates, so equal spacing in displacement is very
        # much not equal spacing in frame number
        n_frames = 100
        xyz = np.zeros((n_frames, 3, 3))
        xyz[:, 0, 0] = np.linspace(0.0, 1.0, n_frames) ** 2

        picks = select_frames_by_msd(xyz, 6)

        assert picks[0] == 0 and picks[-1] == n_frames - 1
        travelled = xyz[picks, 0, 0]
        spacing = np.diff(travelled)
        assert np.allclose(spacing, spacing[0], atol=0.02)


class TestCvFromColvar:
    def test_it_drops_the_row_written_at_step_zero(self, tmp_path: Path) -> None:
        colvar = tmp_path / "COLVAR_SMD"
        cv = np.linspace(0.0, 1.0, 21)
        write_colvar(colvar, cv)

        assert np.allclose(cv_from_colvar(str(colvar), 20, cv_name="pt_cv"), cv[1:])

    def test_it_resamples_a_mismatched_stride(self, tmp_path: Path) -> None:
        colvar = tmp_path / "COLVAR_SMD"
        write_colvar(colvar, np.linspace(0.0, 1.0, 51))

        cv = cv_from_colvar(str(colvar), 10, cv_name="pt_cv")

        assert cv.shape == (10,)
        assert np.isclose(cv[-1], 1.0)
        assert np.all(np.diff(cv) > 0)


class TestEstimatePathLambda:
    def test_it_reports_inverse_square_nanometres_by_default(
        self, tmp_path: Path, pt_pdb: Path
    ) -> None:
        import mdtraj as md

        path_file = tmp_path / "path.pdb"
        fake_steered_traj(path_file, pt_pdb, n_frames=2)

        path = md.load(str(path_file))
        path.superpose(path[0])
        msd_nm2 = np.mean(np.sum((path.xyz[0] - path.xyz[1]) ** 2, axis=1))

        assert estimate_path_lambda(str(path_file)) == pytest.approx(2.3 / msd_nm2)

    def test_angstrom_gives_a_lambda_a_hundred_times_smaller(
        self, tmp_path: Path, pt_pdb: Path
    ) -> None:
        # LAMBDA has units of inverse squared length, so switching from
        # nanometres to angstrom is a factor of 100, not 10.
        path_file = tmp_path / "path.pdb"
        fake_steered_traj(path_file, pt_pdb, n_frames=2)

        in_nm = estimate_path_lambda(str(path_file))
        in_angstrom = estimate_path_lambda(str(path_file), length_unit="A")

        assert in_angstrom == pytest.approx(in_nm / 100.0)

    def test_an_unknown_length_unit_is_rejected(
        self, tmp_path: Path, pt_pdb: Path
    ) -> None:
        path_file = tmp_path / "path.pdb"
        fake_steered_traj(path_file, pt_pdb, n_frames=2)

        with pytest.raises(ValueError, match="Unknown length unit"):
            estimate_path_lambda(str(path_file), length_unit="bohr")

    def test_a_single_frame_path_is_an_error(
        self, tmp_path: Path, pt_pdb: Path
    ) -> None:
        with pytest.raises(ValueError, match="at least two"):
            estimate_path_lambda(str(pt_pdb))


class TestPathFromSteeredMd:
    def test_it_writes_a_pathmsd_reference(self, tmp_path: Path, pt_pdb: Path) -> None:
        import mdtraj as md

        traj_file = tmp_path / "smd_steps.pdb"
        _, fraction = fake_steered_traj(traj_file, pt_pdb, n_frames=60, noise=0.001)
        write_colvar(tmp_path / "COLVAR_SMD", np.concatenate(([0.0], fraction)))

        output = tmp_path / "neb_path.pdb"
        lambda_val = path_from_steered_md(
            str(traj_file),
            template_pdb=str(pt_pdb),
            output_file=str(output),
            colvar_file=str(tmp_path / "COLVAR_SMD"),
            cv_name="pt_cv",
            n_images=8,
            atom_line="ATOM",
        )

        assert lambda_val > 0.0
        path = md.load(str(output))
        assert path.n_frames == 8
        assert path.n_atoms == md.load(str(pt_pdb)).n_atoms
        assert os.path.exists(tmp_path / "neb_path.xyz")

        # The proton should walk from the donor to the acceptor along the path
        to_donor = md.compute_distances(path, [[PT_HYDROGEN, PT_DONOR]]).ravel()
        to_acceptor = md.compute_distances(path, [[PT_HYDROGEN, PT_ACCEPTOR]]).ravel()
        assert to_donor[0] < to_acceptor[0]
        assert to_donor[-1] > to_acceptor[-1]

    def test_it_works_without_a_colvar(self, tmp_path: Path, pt_pdb: Path) -> None:
        import mdtraj as md

        traj_file = tmp_path / "smd_steps.pdb"
        fake_steered_traj(traj_file, pt_pdb, n_frames=40)

        output = tmp_path / "neb_path.pdb"
        path_from_steered_md(
            str(traj_file),
            template_pdb=str(pt_pdb),
            output_file=str(output),
            colvar_file=None,
            n_images=6,
            smooth=2,
            atom_line="ATOM",
        )

        assert md.load(str(output)).n_frames == 6

    def test_it_checks_the_template_matches(self, tmp_path: Path, pt_pdb: Path) -> None:
        traj_file = tmp_path / "smd_steps.pdb"
        fake_steered_traj(traj_file, pt_pdb, n_frames=20)

        with pytest.raises(ValueError, match="atoms but"):
            path_from_steered_md(
                str(traj_file),
                template_pdb=str(pt_pdb),
                output_file=str(tmp_path / "neb_path.pdb"),
                colvar_file=None,
                atom_indices=[PT_DONOR, PT_HYDROGEN, PT_ACCEPTOR],
                n_images=5,
                atom_line="ATOM",
            )

    def test_it_wants_more_frames_than_images(
        self, tmp_path: Path, pt_pdb: Path
    ) -> None:
        traj_file = tmp_path / "smd_steps.pdb"
        fake_steered_traj(traj_file, pt_pdb, n_frames=5)

        with pytest.raises(ValueError, match="too few"):
            path_from_steered_md(
                str(traj_file),
                template_pdb=str(pt_pdb),
                output_file=str(tmp_path / "neb_path.pdb"),
                colvar_file=None,
                n_images=15,
                atom_line="ATOM",
            )

    def test_a_wrong_atom_line_is_caught_rather_than_writing_a_short_path(
        self, tmp_path: Path, pt_pdb: Path
    ) -> None:
        # ASE writes ATOM records; asking for HETATM would silently carry no
        # atoms into the reference, so this must fail loudly.
        traj_file = tmp_path / "smd_steps.pdb"
        fake_steered_traj(traj_file, pt_pdb, n_frames=20)

        with pytest.raises(ValueError, match="records"):
            path_from_steered_md(
                str(traj_file),
                template_pdb=str(pt_pdb),
                output_file=str(tmp_path / "neb_path.pdb"),
                colvar_file=None,
                n_images=5,
                atom_line="HETATM",
            )
