"""
SUNPy A Python Library for solving SU(N) Heisenberg models
Copyright (C) 2026  Samuel Gozel, GNU GPLv3

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pytest
from sunpy.common import partitions



@pytest.mark.parametrize("n, expected_result", [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 5),
    (5, 7),
    (6, 11),
    (7, 15),
    (8, 22),
    (9, 30),
    (10, 42),
    (20, 627),
    (30, 5604)
])
def test_p(n, expected_result):
    assert partitions.p(n) == expected_result
