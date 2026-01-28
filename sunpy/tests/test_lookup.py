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
import numpy as np

from sunpy.sun import sun
from sunpy.lookup import lookup



@pytest.mark.parametrize("N, alpha, nlookupboxes", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(3)
    #:::::::::::::::::::::::::::::::::::::::::::
    (int(3), np.array([3,2,2], dtype=int), int(2)),
    (int(3), np.array([3,2,2], dtype=int), int(3)),
    (int(3), np.array([3,2,2], dtype=int), int(4)),
    (int(3), np.array([3,2,2], dtype=int), int(5)),
    (int(3), np.array([3,2,2], dtype=int), int(6)),
    #------------------------
    # (int(3), np.array([3,3,3], dtype=int), int(2)),
    # (int(3), np.array([3,3,3], dtype=int), int(3)),
    (int(3), np.array([3,3,3], dtype=int), int(4)),
    (int(3), np.array([3,3,3], dtype=int), int(5)),
    (int(3), np.array([3,3,3], dtype=int), int(6)),
    # (int(3), np.array([3,3,3], dtype=int), int(7)),
    # (int(3), np.array([3,3,3], dtype=int), int(8)),
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(4)
    #:::::::::::::::::::::::::::::::::::::::::::
    # (int(4), np.array([4,4,4,4], dtype=int), int(2)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(3)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(4)),
    (int(4), np.array([4,4,4,4], dtype=int), int(5)),
    (int(4), np.array([4,4,4,4], dtype=int), int(6)),
    (int(4), np.array([4,4,4,4], dtype=int), int(7)),
    (int(4), np.array([4,4,4,4], dtype=int), int(8)),
    (int(4), np.array([4,4,4,4], dtype=int), int(9)),
    (int(4), np.array([4,4,4,4], dtype=int), int(10)),
    (int(4), np.array([4,4,4,4], dtype=int), int(11)),
    (int(4), np.array([4,4,4,4], dtype=int), int(12)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(13)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(14)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(15)),
])
def test_partial_lookup_get_SYT(N, alpha, nlookupboxes):
    expected = True
    
    # Generate lookup tool and initialize it
    lkptool = lookup.PartialLookupTool(N, alpha, nlookupboxes)
    lkptool.init_lookup()
    
    # Generate all SYTs in iLLOS
    Y = sun.get_SYT(alpha, order='iLLOS')
    
    # Test
    actual = True
    for i in range(0, Y.shape[0]):
        yi = Y[i]
        y = lkptool.get_SYT(i)
        if not np.sum(abs(y-yi))==0:
            actual = False
            break
    
    assert actual==expected



@pytest.mark.parametrize("N, alpha, nlookupboxes", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(3)
    #:::::::::::::::::::::::::::::::::::::::::::
    (int(3), np.array([3,2,2], dtype=int), int(2)),
    (int(3), np.array([3,2,2], dtype=int), int(3)),
    (int(3), np.array([3,2,2], dtype=int), int(4)),
    (int(3), np.array([3,2,2], dtype=int), int(5)),
    (int(3), np.array([3,2,2], dtype=int), int(6)),
    #------------------------
    # (int(3), np.array([3,3,3], dtype=int), int(2)),
    # (int(3), np.array([3,3,3], dtype=int), int(3)),
    (int(3), np.array([3,3,3], dtype=int), int(4)),
    (int(3), np.array([3,3,3], dtype=int), int(5)),
    (int(3), np.array([3,3,3], dtype=int), int(6)),
    (int(3), np.array([3,3,3], dtype=int), int(7)),
    (int(3), np.array([3,3,3], dtype=int), int(8)),
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(4)
    #:::::::::::::::::::::::::::::::::::::::::::
    # (int(4), np.array([4,4,4,4], dtype=int), int(6)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(7)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(8)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(9)),
    (int(4), np.array([4,4,4,4], dtype=int), int(10)),
    (int(4), np.array([4,4,4,4], dtype=int), int(11)),
    (int(4), np.array([4,4,4,4], dtype=int), int(12)),
    (int(4), np.array([4,4,4,4], dtype=int), int(13)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(14)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(15)),
])
def test_partial_lookup_get_index(N, alpha, nlookupboxes):
    expected = True
    
    # Generate lookup tool and initialize it
    lkptool = lookup.PartialLookupTool(N, alpha, nlookupboxes)
    lkptool.init_lookup()
    
    # Generate all SYTs in iLLOS
    Y = sun.get_SYT(alpha, order='iLLOS')
    
    # Test
    actual = True
    for i in range(0, Y.shape[0]):
        ii = lkptool.get_index(Y[i])
        if not ii==i:
            actual = False
            break
    
    assert actual==expected
