"""
SUN4Py A Python Library for solving SU(N) Heisenberg models
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
import pickle
import os

from sun4py.sun import sun



prec = 1.0e-13



@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), np.array([[0]], dtype=int)),
    (np.array([2], dtype=int), np.array([[0, 0]], dtype=int)),
    (np.array([1,1], dtype=int), np.array([[0, 1]], dtype=int)),
    (np.array([2,1], dtype=int), np.array([[0, 0, 1], [0, 1, 0]], dtype=int))
])
def test_get_SYT(alpha, expected):
    assert(np.linalg.norm( sun.get_SYT(alpha) - expected )==0)


@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), int(1)),
    (np.array([2], dtype=int), int(1)),
    (np.array([1, 1], dtype=int), int(1)),
    (np.array([2, 1], dtype=int), int(2)),
    # See Nataf & Mila, PRL 113, 127204 (2014) for the 4 following dimensions
    (np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=int), int(1430)),
    (np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=int), int(16796)),
    (np.array([4, 4, 4, 4, 4], dtype=int), int(1662804)),
    (np.array([5, 5, 5, 5, 5], dtype=int), int(701149020))
])
def test_multiplicity(alpha, expected):
    assert(sun.multiplicity(alpha)==expected)


@pytest.mark.parametrize("alpha, m, expected", [
    # See Tables I & II of Nataf & Mila, Phys. Rev. B 93, 155134 (2016)
    # -- m=2 --
    (np.array([10, 10, 10], dtype=int), int(2), int(6879236)),
    (np.array([11, 10, 9], dtype=int), int(2), int(44994040)),
    (np.array([12, 12, 12], dtype=int), int(2), int(767746656)),
    (np.array([8, 8, 8, 8], dtype=int), int(2), int(190720530)),
    (np.array([9, 8, 8, 7], dtype=int), int(2), int(2077175100)),
    (np.array([6, 6, 6, 6, 6], dtype=int), int(2), int(25468729)),
    (np.array([7, 6, 6, 6, 5], dtype=int), int(2), int(377182806)),
    (np.array([4, 4, 4, 4, 4, 4], dtype=int), int(2), int(16071)),
    (np.array([5, 4, 4, 4, 4, 3], dtype=int), int(2), int(272712)),
    (np.array([4, 4, 4, 4, 4, 4, 4, 4], dtype=int), int(2), int(3607890)),
    (np.array([5, 4, 4, 4, 4, 4, 4, 3], dtype=int), int(2), int(93683590)),
    #(np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=int), int(2), int(1135871490)),
    # -- m=3 --
    (np.array([12, 12, 12], dtype=int), int(3), int(3463075)),
    (np.array([9, 9, 9, 9], dtype=int), int(3), int(10260228)),
    (np.array([6, 6, 6, 6, 6, 6], dtype=int), int(3), int(1113860))
])
def test_multiplicity_symm(alpha, m, expected):
    assert(sun.multiplicity_symm(alpha, m)==expected)


@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), 0),
    (np.array([2], dtype=int), 1),
    (np.array([1,1], dtype=int), -1),
    (np.array([3], dtype=int), 3),
    (np.array([2,1], dtype=int), 0),
    (np.array([1,1,1], dtype=int), -3),
    (np.array([4], dtype=int), 6),
    (np.array([3,1], dtype=int), 2),
    (np.array([2,2], dtype=int), 0),
    (np.array([2,1,1], dtype=int), -2),
    (np.array([1,1,1,1], dtype=int), -6),
    (np.array([5], dtype=int), 10),
    (np.array([4,1], dtype=int), 5),
    (np.array([3,2], dtype=int), 2),
    (np.array([3,1,1], dtype=int), 0),
    (np.array([2,2,1], dtype=int), -2),
    (np.array([2,1,1,1], dtype=int), -5),
    (np.array([1,1,1,1,1], dtype=int), -10)
])
def test_casimir(alpha, expected):
    assert(abs( sun.casimir_quadratic(alpha) - expected )<prec)


@pytest.mark.parametrize("alpha, N, expected", [
    (np.array([1, 0, 0]), int(3), int(3)), 
    (np.array([1, 1, 0]), int(3), int(3)), 
    (np.array([1, 1, 1]), int(3), int(1)), 
    (np.array([2, 1, 0]), int(3), int(8)), 
    (np.array([2, 2, 0]), int(3), int(6)), 
    (np.array([3, 0, 0]), int(3), int(10)), 
    (np.array([3, 3, 0]), int(3), int(10)), 
    (np.array([4, 2, 0]), int(3), int(27)), 
    (np.array([2, 1, 1, 0]), int(4), int(15)), 
    (np.array([2, 1, 1, 1, 0]), int(5), int(24)), 
    (np.array([4, 3, 2, 1, 0]), int(5), int(1024))
])
def test_dim_irrep_sun(alpha, N, expected):
    dimension = sun.dim_irrep_sun(alpha, N)
    assert(dimension==expected)


@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), True),
    (np.array([2], dtype=int), True),
    (np.array([1,1], dtype=int), True),
    (np.array([3], dtype=int), True),
    (np.array([2,1], dtype=int), True),
    (np.array([1,1,1], dtype=int), True),
    (np.array([3,3,3,3], dtype=int), True)
])
def test_binary_search_SYT(alpha, expected):
    actual = True
    # extracts all SYTs
    Y = sun.get_SYT(alpha=alpha, order='iLLOS')
    # test for each SYT
    for i in range(0, Y.shape[0]):
        ii = sun.binary_search_SYT(Y[i,:], Y)
        if not ii==i:
            actual = False
            break
    assert(actual==expected)


@pytest.mark.parametrize("alpha, m, order, expected", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - LLOS
    (np.array([1], dtype=int), int(1), 'LLOS', True),
    (np.array([2], dtype=int), int(1), 'LLOS', True),
    (np.array([1,1], dtype=int), int(1), 'LLOS', True),
    (np.array([3], dtype=int), int(1), 'LLOS', True),
    (np.array([2,1], dtype=int), int(1), 'LLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'LLOS', True),
    # (np.array([3,3,3,3], dtype=int), int(1), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - iLLOS
    (np.array([1], dtype=int), int(1), 'iLLOS', True),
    (np.array([2], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([3], dtype=int), int(1), 'iLLOS', True),
    (np.array([2,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'iLLOS', True),
    # (np.array([3,3,3,3], dtype=int), int(1), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - LLOS
    (np.array([2,2,2], dtype=int), int(2), 'LLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'LLOS', True),
    # (np.array([6,6,2], dtype=int), int(2), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - iLLOS
    (np.array([2,2,2], dtype=int), int(2), 'iLLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'iLLOS', True),
    # (np.array([6,6,2], dtype=int), int(2), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - LLOS
    (np.array([3,3,3], dtype=int), int(3), 'LLOS', True),
    # (np.array([5,5,5], dtype=int), int(3), 'LLOS', True),
    # (np.array([6,5,4], dtype=int), int(3), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - iLLOS
    (np.array([3,3,3], dtype=int), int(3), 'iLLOS', True),
    # (np.array([5,5,5], dtype=int), int(3), 'iLLOS', True),
    # (np.array([6,5,4], dtype=int), int(3), 'iLLOS', True),
])
def test_map_syt_to_index(alpha, m, order, expected):
    actual = True
    # extracts all SYTs
    if m==1:
        Y = sun.get_SYT(alpha=alpha, order=order)
    else:
        Y = sun.get_SYT_symm(alpha=alpha, m=m, order=order)
    # check for each SYT
    for i in range(0, Y.shape[0]):
        if m==1:
            ii = sun.SYT_to_index(Y[i,:], alpha, order)
        else:
            ii = sun.SYT_to_index_symm(Y[i,:], alpha, m, order)
    
        if not ii==i:
            actual = False
            break
    assert(actual==expected)


@pytest.mark.parametrize("alpha, m, order, expected", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - LLOS
    (np.array([1], dtype=int), int(1), 'LLOS', True),
    (np.array([2], dtype=int), int(1), 'LLOS', True),
    (np.array([1,1], dtype=int), int(1), 'LLOS', True),
    (np.array([3], dtype=int), int(1), 'LLOS', True),
    (np.array([2,1], dtype=int), int(1), 'LLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'LLOS', True),
    # (np.array([3,3,3,3], dtype=int), int(1), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - iLLOS
    (np.array([1], dtype=int), int(1), 'iLLOS', True),
    (np.array([2], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([3], dtype=int), int(1), 'iLLOS', True),
    (np.array([2,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'iLLOS', True),
    # (np.array([3,3,3,3], dtype=int), int(1), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - LLOS
    (np.array([2,2,2], dtype=int), int(2), 'LLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'LLOS', True),
    # (np.array([6,6,2], dtype=int), int(2), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - iLLOS
    (np.array([2,2,2], dtype=int), int(2), 'iLLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'iLLOS', True),
    # (np.array([6,6,2], dtype=int), int(2), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - LLOS
    (np.array([3,3,3], dtype=int), int(3), 'LLOS', True),
    # (np.array([5,5,5], dtype=int), int(3), 'LLOS', True),
    # (np.array([6,5,4], dtype=int), int(3), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - iLLOS
    (np.array([3,3,3], dtype=int), int(3), 'iLLOS', True),
    # (np.array([5,5,5], dtype=int), int(3), 'iLLOS', True),
    # (np.array([6,5,4], dtype=int), int(3), 'iLLOS', True),
])
def test_map_index_to_syt(alpha, m, order, expected):
    actual = True
    # extracts all SYTs
    if m==1:
        Y = sun.get_SYT(alpha, order=order)
    else:
        Y = sun.get_SYT_symm(alpha, m, order=order)
    
    # check for each SYT
    for i in range(0, Y.shape[0]):
        if m==1:
            yi = sun.index_to_SYT(i, alpha, order)
        else:
            yi = sun.index_to_SYT_symm(i, alpha, m, order)
        
        if np.linalg.norm(Y[i,:]-yi)>0:
            actual = False
            break
    
    assert(actual==expected)


@pytest.mark.parametrize("y, ytarget, expected", [
    (np.array([0, 0, 1, 2], dtype=int), np.array([0, 1, 0, 2], dtype=int), (np.array([1], dtype=int), np.array([1./2.]))),
    (np.array([0, 1, 2, 0], dtype=int), np.array([0, 0, 1, 2], dtype=int), (np.array([2, 1], dtype=int), np.array([-1./3., -1./2.])))
])
def test_SYT_to_target_1(y, ytarget, expected):
    # arange
    # ...
    # act
    sigma, rho = sun.SYT_to_target(y, ytarget)
    # assert
    assert(np.linalg.norm( sigma - expected[0] )==0)
    assert(np.linalg.norm( rho - expected[1] )<prec)


@pytest.mark.parametrize("y, ytarget", [
    (np.array([0, 1, 0, 2], dtype=int), np.array([0, 0, 1, 2], dtype=int)), 
    (np.array([0, 0, 1, 2], dtype=int), np.array([0, 1, 2, 0], dtype=int))
])
def test_SYT_to_target_2(y, ytarget):
    # arange
    # ...
    # act
    sigma, rho = sun.SYT_to_target(y, ytarget)
    sigmap, rhop = sun.SYT_to_target(ytarget, y)
    # assert
    assert(np.linalg.norm(sigma - np.flip(sigmap))==0)
    assert(np.linalg.norm(rho - np.flip(-rhop))<prec)


@pytest.mark.parametrize("n", [
    (int(1)), 
    (int(2)), 
    (int(3)), 
    (int(10))
])
def test_get_all_irreps_SU2(n):
    # arange
    N = int(2)
    # act
    irreps = sun.get_all_irreps(N, n)
    # assert
    assert(np.sum(abs(irreps[:, 0] - np.arange(0, n+1)))==0)
    assert(np.sum(abs(irreps[:, 1]))==0)

@pytest.mark.parametrize("N, n", [
    #-------------
    # SU(3)
    (int(3), int(1)), 
    (int(3), int(2)), 
    (int(3), int(3)), 
    (int(3), int(4)), 
    (int(3), int(5)),
    #-------------
    # SU(4)
    (int(4), int(1)), 
    (int(4), int(2)), 
    (int(4), int(3)), 
    (int(4), int(4)), 
    (int(4), int(5)), 
    #-------------
    # SU(5)
    (int(5), int(1)), 
    (int(5), int(2)), 
    (int(5), int(3)), 
    (int(5), int(4)), 
    (int(5), int(5))
])
def test_get_all_irreps_SU(N, n):
    # arange
    test_dir = os.path.dirname(os.path.abspath(__file__))
    testdata_file = f'testdata_irreps_SU{N}.pickle'
    testdata_dir = 'testdata'
    test_datafile = os.path.join(test_dir, testdata_dir, testdata_file)
    with open(test_datafile, 'rb') as file:
        data = pickle.load(file)
        expected = data[n]
    # act
    irreps = sun.get_all_irreps(N, n)
    # assert
    assert(np.sum(abs(irreps - expected))==0)
