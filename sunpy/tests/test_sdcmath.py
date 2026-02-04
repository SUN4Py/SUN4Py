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

import numpy as np

from sunpy.sdc.math import sdcmath

prec = 1.0e-13



def test_SDC_1_418_2a_1():
    """
    1st line of Table 4.18 2a page 187 of
        Group Representation Theory for Physicists
        Jin-Quan Chen, Hialun Ping and Fan Wang
        World Scientific, 2nd edition, (2002)
    """
    # arrange
    N = int(3)
    nu = np.array([4, 1, 0], dtype=int)
    nu1 = np.array([1, 0, 0], dtype=int)
    nu2 = np.array([4, 0, 0], dtype=int)
    
    expected_coeff0 = np.sqrt(np.array([3, 5, 10, 30])/48.0) # line 1 of Table 4.18 2a
    
    # act
    _, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2)
    
    # assert
    assert(np.sum(abs(sdcs[0, :, 0] - expected_coeff0))<prec)
    

def test_SDC_2_418_2a_234():
    """
    2nd, 3rd and 4th lines of Table 4.18 2a page 187 of
        Group Representation Theory for Physicists
        Jin-Quan Chen, Hialun Ping and Fan Wang
        World Scientific, 2nd edition, (2002)
    """
    # arrange
    N = int(3)
    nu = np.array([4, 1, 0], dtype=int)
    nu1 = np.array([1, 0, 0], dtype=int)
    nu2 = np.array([3, 1, 0], dtype=int)
    
    expected_coeff0 = np.sqrt(np.array([135, 1, 2, 6])/144.0) # line 2 of Table 4.18 2a
    expected_coeff0[[1, 2, 3]] *= -1
    expected_coeff1 = np.sqrt(np.array([0, 32, 1, 3])/36.0) # line 3 of Table 4.18 2a
    expected_coeff1[[2, 3]] *= -1
    expected_coeff2 = np.sqrt(np.array([0, 0, 3, 1])/4.0) # line 4 of Table 4.18 2a
    expected_coeff2[3] *= -1
    
    # act
    _Y, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2)
    
    # assert
    assert(np.sum(abs(sdcs[0, :, 0] - expected_coeff0))<prec)
    assert(np.sum(abs(sdcs[1, :, 0] - expected_coeff1))<prec)
    assert(np.sum(abs(sdcs[2, :, 0] - expected_coeff2))<prec)


def test_SDC_3_418_2a_5():
    """
    5th line of Table 4.18 2a page 187 of
        Group Representation Theory for Physicists
        Jin-Quan Chen, Hialun Ping and Fan Wang
        World Scientific, 2nd edition, (2002)
    """
    # arrange
    N = int(3)
    nu = np.array([4, 1, 0], dtype=int)
    nu1 = np.array([2, 0, 0], dtype=int)
    nu2 = np.array([3, 0, 0], dtype=int)
    
    expected_coeff0 = np.sqrt(np.array([3, 5, 10])/18.0)
    
    # act
    _, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2)
    
    # assert
    assert(np.sum(abs(sdcs[0, :, 0] - expected_coeff0))<prec)


def test_SDC_4_418_2a_67():
    """
    6th and 7th lines of Table 4.18 2a page 187 of
        Group Representation Theory for Physicists
        Jin-Quan Chen, Hialun Ping and Fan Wang
        World Scientific, 2nd edition, (2002)
    """
    # arrange
    N = int(3)
    nu = np.array([4, 1, 0], dtype=int)
    nu1 = np.array([2, 0, 0], dtype=int)
    nu2 = np.array([2, 1, 0], dtype=int)
    
    expected_coeff0 = np.sqrt(np.array([15, 1, 2])/18.0)
    expected_coeff0[[1, 2]] *= -1
    expected_coeff1 = np.sqrt(np.array([0, 2, 1])/3.0)
    expected_coeff1[2] *= -1
    
    # act
    _, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2)
    
    # assert
    assert(np.sum(abs(sdcs[0, :, 0] - expected_coeff0))<prec)
    assert(np.sum(abs(sdcs[1, :, 0] - expected_coeff1))<prec)


def test_SDC_5_418_2a_67_prime():
    """
    6th and 7th lines of Table 4.18 2a page 187 of
        Group Representation Theory for Physicists
        Jin-Quan Chen, Hialun Ping and Fan Wang
        World Scientific, 2nd edition, (2002)
    """
    
    # arrange
    N = int(3)
    nu = np.array([4, 1, 0], dtype=int)
    nu1 = np.array([2, 0, 0], dtype=int)
    nu2 = np.array([2, 1, 0], dtype=int)
    y2_0 = np.array([0, 0, 1], dtype=int)
    y2_1 = np.array([0, 1, 0], dtype=int)
    
    expected_coeff0 = np.sqrt(np.array([15, 1, 2])/18.0)
    expected_coeff0[[1, 2]] *= -1
    expected_coeff1 = np.sqrt(np.array([0, 2, 1])/3.0)
    expected_coeff1[2] *= -1
    
    # act
    _, _, sdcs0 = sdcmath.get_SDC(N, nu, nu1, nu2, Y2=y2_0)
    _, _, sdcs1 = sdcmath.get_SDC(N, nu, nu1, nu2, Y2=y2_1)
    
    # assert
    assert(np.sum(abs(sdcs0[0, :, 0] - expected_coeff0))<prec)
    assert(np.sum(abs(sdcs1[0, :, 0] - expected_coeff1))<prec)


def test_SDC_6_418_2c_123():
    """
    1st, 2nd and 3rd lines of Table 4.18 2c page 187 of
        Group Representation Theory for Physicists
        Jin-Quan Chen, Hialun Ping and Fan Wang
        World Scientific, 2nd edition, (2002)
    """
    
    # arrange
    N = int(3)
    nu = np.array([3, 1, 1], dtype=int)
    nu1 = np.array([1, 0, 0], dtype=int)
    nu2 = np.array([3, 1, 0], dtype=int)
    
    expected_coeff0 = np.sqrt(np.array([1, 2, 6, 0, 0, 0])/9.0)
    expected_coeff1 = np.sqrt(np.array([32, 1, 3, 135, 405, 0])/576.0)
    expected_coeff1[0] *= -1
    expected_coeff2 = np.sqrt(np.array([0, 9, 3, 15, 5, 160])/192.0)
    expected_coeff2[[1, 3]] *= -1
    
    # act
    _, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2)
    
    # assert
    assert(np.sum(abs(sdcs[0, :, 0] - expected_coeff0))<prec)
    assert(np.sum(abs(sdcs[1, :, 0] - expected_coeff1))<prec)
    assert(np.sum(abs(sdcs[2, :, 0] - expected_coeff2))<prec)


def test_SDC_7_418_2c_456():
    """
    4th, 5th and 6th lines of Table 4.18 2c page 187 of
        Group Representation Theory for Physicists
        Jin-Quan Chen, Hialun Ping and Fan Wang
        World Scientific, 2nd edition, (2002)
    """
    # arrange
    N = int(3)
    nu = np.array([3, 1, 1], dtype=int)
    nu1 = np.array([1, 0, 0], dtype=int)
    nu2 = np.array([2, 1, 1], dtype=int)
    
    expected_coeff0 = np.sqrt(np.array([160, 5, 15, 3, 9, 0])/192.0)
    expected_coeff0[[1, 2]] *= -1
    expected_coeff1 = np.sqrt(np.array([0, 405, 135, 3, 1, 32])/576.0)
    expected_coeff1[[2, 3]] *= -1 # likely a typo in Chen's book: no minus sign on sqrt(405/576)
    expected_coeff2 = np.sqrt(np.array([0, 0, 0, 6, 2, 1])/9.0)
    expected_coeff2[4] *= -1
    
    # act
    _, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2)
    
    # assert
    assert(np.sum(abs(sdcs[0, :, 0] - expected_coeff0))<prec)
    assert(np.sum(abs(sdcs[1, :, 0] - expected_coeff1))<prec)
    assert(np.sum(abs(sdcs[2, :, 0] - expected_coeff2))<prec)


def test_SDC_8():
    """
    See Eq. (G.8), (G.14), (G.19) of:
        Asymptotic freedom, Haldane gap and edge states of SU(N) spin chains
        Samuel Gozel
        EPFL, 2020
        https://doi.org/10.5075/epfl-thesis-8417
    """
    # arrange
    N = int(3)
    nu = np.array([4, 4, 3], dtype=int)
    nu1 = np.array([3, 3, 1], dtype=int)
    nu2 = np.array([2, 1, 1], dtype=int)
    
    expected_coeff0 = np.array([0.0, np.sqrt(15)/8, 5./8., -np.sqrt(5)/8, 
                                -5./(8*np.sqrt(3)), 1./np.sqrt(6)])
    expected_coeff1 = np.array([np.sqrt(5/2)/3, 5*np.sqrt(5)/24, -5./(8*np.sqrt(3)),
                                -np.sqrt(15)/8, 3.0/8, 0.0])
    expected_coeff2 = np.array([np.sqrt(5)/3, -np.sqrt(5/2)/3, 1./np.sqrt(6),
                                0.0, 0.0, 0.0])
    # act
    _, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2, ref2firstLLOS=True, ref1firstLLOS=True)
    
    # assert
    assert(np.linalg.norm(sdcs[0, :, 0] - expected_coeff0)<prec)
    assert(np.linalg.norm(sdcs[1, :, 0] - expected_coeff1)<prec)
    assert(np.linalg.norm(sdcs[2, :, 0] - expected_coeff2)<prec)


def test_SDC_9():
    """
    See Table II.23 of:
        Jin-Quan Chen, David F. Collinson, Mei-Juan Gao
        J. Math. Phys. 24, 2695–2705 (1983)
        https://doi.org/10.1063/1.525668
    
    See also:
    Section 3.1.8 of:
        Asymptotic freedom, Haldane gap and edge states of SU(N) spin chains
        Samuel Gozel
        EPFL, 2020
        https://doi.org/10.5075/epfl-thesis-8417
    """
    # arrange
    N = int(3)
    nu = np.array([3, 2, 1], dtype=int)
    nu1 = np.array([2, 1, 0], dtype=int)
    nu2 = np.array([2, 1, 0], dtype=int)
    Ntau = int(2)
    NY2 = int(2)
    NY = int(6)
    
    expected = np.zeros(shape=(NY2, NY, Ntau), dtype=float)
    expected[0, :, 0] = np.array([np.sqrt(1/6), np.sqrt(1/2), 
                                -np.sqrt(1/8), -np.sqrt(5/24), 0.0, 0.0])
    expected[1, :, 0] = np.array([-np.sqrt(1/8), np.sqrt(1/24), 
                                 0.0, 0.0, np.sqrt(5/8), -np.sqrt(5/24)])
    expected[0, :, 1] = np.array([np.sqrt(5/96), np.sqrt(5/32), 
                                np.sqrt(5/32), np.sqrt(25/96), -np.sqrt(3/32), -np.sqrt(9/32)])
    expected[1, :, 1] = np.array([np.sqrt(5/32), -np.sqrt(5/96), 
                                np.sqrt(15/32), -np.sqrt(9/32), np.sqrt(1/32), -np.sqrt(1/96)])
    
    # act
    _, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2, ref2firstLLOS=True, ref1firstLLOS=True)
    
    # assert
    assert(sdcs.shape[0]==NY2)
    assert(sdcs.shape[1]==NY)
    assert(sdcs.shape[2]==Ntau)
    
    for tau in range(Ntau):
        for m2 in range(NY2):
            assert(np.sum(abs(sdcs[m2, :, tau] - expected[m2, :, tau]))/NY<prec)
    