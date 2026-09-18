"""Geodesic interpolation for reaction paths.

Builds a smooth initial guess for a reaction path between two or more
geometries by finding the shortest path under a metric of scaled inter-atomic
distances. ``geodesic_interpolate`` is the entry point; the XYZ and ASE helpers
are re-exported for convenience.

Vendored code
-------------
This subpackage is a copy of
https://github.com/LouieSlocombe/geodesic_interpolate, itself a fork of
https://github.com/virtualzx-nad/geodesic-interpolate by Xiaolei Zhu. It is MIT
licensed; the licence and copyright notice are kept beside this file in
``LICENSE`` and travel with every distribution.

It is vendored rather than depended on because the fork is not on PyPI, which
rejects distributions carrying a direct URL requirement, and the version that
*is* on PyPI under ``geodesic-interpolate`` exposes only ``Geodesic`` and
``redistribute`` -- not the ASE-aware ``geodesic_interpolate`` entry point, the
seeded random generator threaded through the interpolation, or ``align_path_to``.

The subpackage is private: import it through the public
``reactiontools.quick_guess_path`` and ``prepare_neb(geo_int=True)`` rather than
reaching in here, as its contents track upstream rather than this package's own
API promises. Cite ``zhu2019geodesic`` when you use it; see ``CITATIONS.bib``.
"""

from .coord_utils import align_path_to
from .fileio import from_ase_atoms, read_xyz, to_ase_atoms, write_xyz
from .geodesic import Geodesic
from .interpolation import redistribute
from .main import geodesic_interpolate

__all__ = [
    "Geodesic",
    "align_path_to",
    "from_ase_atoms",
    "geodesic_interpolate",
    "read_xyz",
    "redistribute",
    "to_ase_atoms",
    "write_xyz",
]
