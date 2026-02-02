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

from sunpy.sun import sun



def get_states_fund(N, num_irreps, irreps):
    """
    Create the ensemble of all states (irrep, bottom corner) for a given ensemble
    of input irreps
    
    Parameters
    ----------
    N : int
        SU(N)
    num_irreps : int
        number of irreps
    irreps : numpy array
        array of num_irreps first irreps of SU(N)
    """
    assert(irreps.shape[0]==num_irreps)
    
    states = np.zeros(shape=(N*num_irreps, 2), dtype=int)
    cpt = int(0)
    for p in range(0, num_irreps):
        bc_vec = sun.get_bottom_corner(irreps[p])
        for q in range(0, len(bc_vec)):
            states[cpt+q, 0] = p
            states[cpt+q, 1] = bc_vec[q]
        cpt += len(bc_vec)
    
    states = states[0:cpt]
    
    return states


def get_states(N, num_irreps, irreps, m=1, **kwargs):
    """
    Create the ensemble of all states (irrep, bottom corner) for a given ensemble
    of input irreps
    
    Parameters
    ----------
    N : int
        SU(N)
    num_irreps : int
        number of irreps
    irreps : numpy array
        array of num_irreps first irreps of SU(N)
    m : int [optional][default: 1]
        number of particles per site
    symmetry : str [required when m>1]
        'symmetric' or 'antisymmetric'
    
    Returns
    -------
    states : numpy array
        list of states (stored in the rows) |alpha1, l1>
    
    Details
    -------
    states[i] is the i-th state, and is an array of dimension m+1
        states[i][0] is the index to an irrep in the input <irreps>
        states[i][1:] are the positions of the m bottom corners
    
    """
    assert(irreps.shape[0]==num_irreps)
    
    if m==1:
        return get_states_fund(N, num_irreps, irreps)
    
    states = np.zeros(shape=(N**m * num_irreps, m+1), dtype=int)
    cpt = int(0)
    for p in range(0, num_irreps):
        _, bc_vec = sun.get_ascendants(irreps[p], N, m=m, **kwargs)
        for q in range(0, bc_vec.shape[0]):
            states[cpt+q, 0] = p
            states[cpt+q, 1:] = bc_vec[q]
        cpt += bc_vec.shape[0]
    
    states = states[0:cpt]
    
    return states


def states_rme(N, num_irreps, irreps, m=1, **kwargs):
    """
    Create the ensemble of all states (irrep1, bottom corner1; irrep2, bottom corner2) 
    for a given ensemble of input irreps
    
    Parameters
    ----------
    N : int
        SU(N)
    num_irreps : int
        number of irreps
    irreps : numpy array
        array of num_irreps first irreps of SU(N)
    m : int [optional][default: 1]
        number of particles per site
    symmetry : str [required when m>1]
        'symmetric' or 'antisymmetric'
    
    Returns
    -------
    states : numpy array
        list of "double states" |alpha1, l1; alpha2, l2>
    """
    assert(irreps.shape[0]==num_irreps)
    
    states1 = get_states(N, num_irreps, irreps, m=m, **kwargs)
    num_states = states1.shape[0]
    num_states2 = num_states * num_states
    
    states = np.zeros(shape=(num_states2, 2*(m+1)), dtype=int)
    
    states[:, 0:(m+1)] = np.repeat(states1, repeats=num_states, axis=0)
    states[:, (m+1):] = np.tile(states1, (num_states, 1))
    
    return states
