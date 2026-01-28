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
import scipy.sparse

from sunpy.sun import sun
from sunpy.sdc.common import sdcutils



def get_SDC(N, nu, nu1, l1, nu2, l2, ref2firstLLOS=True, ref1firstLLOS=True):
    """
    Compute the permutational Subduction Coefficients (SDCs) in the convention 
    useful for DMRG.
    
    Parameters
    ----------
    N : int
        SU(N)
    nu : numpy array
        global irrep with n particles
    nu1 : numpy array
        irrep with n1 particles
    l1 : int or numpy array
        position(s) of bottom corner(s) of last particle(s) in nu1
    nu2 : numpy array
        irrep with n2 particles, n=n1+n2
    l2 : int or numpy array
        position(s) of bottom corner(s) of last particle(s) in nu2
    ref1firstLLOS : bool
        reference SYT used for y1
    ref2firstLLOS : bool
        if True, the first SYT in the ascending order of the LLOS is used as 
        reference for fixing the overall phase convention.  If False, the last 
        SYT is used (namely the first in the descending order of LLOS)
    
    Returns
    -------
    Y : numpy array
        SYTs for the irrep nu, with particle n1-1 situated at bottom corner l1 
        and n2 last particles satisfying symmetries imposed by (nu2, l2)
    CY : numpy array
        associated column positions
    COEFF_FINAL : numpy array
        SDCs. COEFF_FINAL is of dimension (NY, Ntau) where
            NY   : is the number of SYTs in the output Y
            Ntau : is the multiplicity of nu in the tensor product of nu1 with nu2,
                   defining two sequences of SDCs
    
    Remarks
    -------
    - ref1firstLLOS has no influence on the SDCs. It does have an influence on
      the SYTs in the output Y.
    
    """
    
    assert len(nu)<=N
    assert len(nu1)<=N
    assert len(nu2)<=N
    
    # ensure irreps have exactly N rows
    if len(nu)<N:
        nu = np.hstack((nu, np.zeros(N-len(nu), dtype=int)))
    if len(nu)<N:
        nu1 = np.hstack((nu1, np.zeros(N-len(nu1), dtype=int)))
    if len(nu2)<N:
        nu2 = np.hstack((nu2, np.zeros(N-len(nu2), dtype=int)))
    
    n = np.sum(nu)
    n1 = np.sum(nu1)
    n2 = np.sum(nu2)
    assert n==n1+n2
    
    nu1nu2nu = sun.multiplicity_irrep_mixed(nu, np.array([nu1, nu2], dtype=int), N)
    
    if not isinstance(l1, np.ndarray):
        l1 = np.array([l1])
    if not isinstance(l2, np.ndarray):
        l2 = np.array([l2])
    
    l1 = np.sort(l1)
    l2 = np.sort(l2)
    assert(len(l1)==len(l2))
    
    # Construct tableau S1 with the last number situated at bottom corner l1
    # Remark: the exact form of S1 is not important. The two only important
    # things are:
    # (1) it is a valid SYT
    # (2) the last number is situated in the bottom corner at row l1
    # 
    # In particular, that means that all other particles can be placed
    # arbitrarily, as long as they satisfy (1) above.
    
    # One possible choice: First SYT in LLOS for particles 0 -> n1-2
    nu1p = np.copy(nu1)
    for bc in l1:
        nu1p[bc] -= 1
    
    if ref1firstLLOS==True:
        y1 = sun.index_to_SYT(0, alpha=nu1p, order='LLOS')
    else:
        y1 = sun.index_to_SYT(0, alpha=nu1p, order='iLLOS')
    y1 = np.hstack((y1, l1))
    
    # construct reference SYT for nu2
    if ref2firstLLOS==True:
        # the reference is the first SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='LLOS')
    else:
        # the reference is the last SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='iLLOS')

    # Construct all SYTs associated to the shape nu-nu1, in increasing order of LLOS
    sY = sun.get_subSYT(nu, nu1, order='LLOS')
    NY = sY.shape[0]

    # Fill all obtained SYTs with numbers from 1 to n1
    Y = np.zeros(shape=(NY, n), dtype=int)
    Y[:, :n1] = np.matlib.repmat(y1, NY, 1)
    Y[:, n1:] = sY
    #Y = np.hstack((np.matlib.repmat(y1, NY, 1), Y))
    CY = sun.get_column(Y)
    
    # Compute the matrices of the adjacent transpositions for S_{n_2}
    MatAdjaTranspo = []
    for k in range(n1, n-1):
        MatAdjaTranspo.append( sun.get_adjacent_transposition_matrix(nu, Y, CY, k) )
    
    # Compute Casimir (eigenvalue of CSCO-II) for the canonical chain
    # associated to S_{n_2}
    casimir = sdcutils.casimir_canonical_chain(nu2, y2_ref)
    
    # construct projection operator
    M = sdcutils.build_canonical_chain_projector(NY, n, n1, MatAdjaTranspo, 
                                                 casimir, ordering='dmrg')
    
    # compute kernel of projection operator
    if M.shape[0]==1:
        assert abs(M[0,0])<1.0e-12
        COEFF_REF = np.array([[1.0]], dtype=float)
        DD = np.array([0.0], dtype=float)
    elif M.shape[0]<100:
        DD, COEFF_REF = np.linalg.eigh(M.todense())
        COEFF_REF = np.asarray(COEFF_REF)
    else:
        DD, COEFF_REF = scipy.sparse.linalg.eigsh(M, k=min(4, NY), which='SA')
    
    # keep only eigenvectors with eigenvalue==0
    ind = np.argwhere(abs(DD)<1.0e-12).flatten()
    assert len(ind)==nu1nu2nu
    COEFF_REF = COEFF_REF[:, ind]
    
    # Fix the overall phase
    COEFF_REF, IND_REF = sdcutils.set_overall_phase(COEFF_REF)
    
    deal_with_phase = False
    for i in range(len(l2)-1, -1, -1):
        l2el = l2[i]
        if (abs(l2el - y2_ref[n2+i-len(l2)])>0):
            deal_with_phase = True
            break
    
    # Compute SDCs
    out = []
    
    if deal_with_phase==0:
        # the last particle in y2_ref is situated at bottom corner l2 in nu2, 
        # thus no need to do any more phase fixing
        #COEFF_FINAL = COEFF_REF
        
        for tau in range(0, nu1nu2nu):
            out.append([Y[IND_REF[:, tau]], CY[IND_REF[:, tau]], COEFF_REF[IND_REF[:, tau], tau]])
        
    else:
        # one needs to find the sequence of transpositions which brings 
        # local particle n2-1 (namely n-1 in global numbering) at row l2 in nu2
        
        # construct SYT y2 with bottom corner at l2
        nu2p = np.copy(nu2)
        for bc in l2:
            nu2p[bc] -= 1
        
        if ref2firstLLOS==True:
            y2 = sun.index_to_SYT(0, alpha=nu2p, order='LLOS')
        else:
            y2 = sun.index_to_SYT(0, alpha=nu2p, order='iLLOS')
        y2 = np.hstack((y2, l2))
        
        # find sequence of transpositions which brings y2_ref to y2
        sigma, rho = sun.SYT_to_target(y2_ref, y2)
        
        # y2 = sigma[-1] sigma[-2] ... sigma[1] sigma[0] y2_ref
        
        # IMPORTANT CORRESPONDANCE
        # y2_ref[i] == row of number i, 0<=i<n2
        # <==>
        # y2_ref[i] == row of number n-i-1, 0<=i<n2 in the global SYT
        # same for y2
        
        # go from local to global numbering
        sigma = n - sigma - 2
        
        #COEFF_FINAL = np.zeros(shape=COEFF_REF.shape, dtype=float)
        
        # apply the sequence of transpositions
        
        for tau in range(0, nu1nu2nu):
            ydev1, cydev1, coeff1 = sun.apply_transpositions(
                                            nu, 
                                            sigma, 
                                            rho, 
                                            np.copy(Y[IND_REF[:, tau]]), 
                                            np.copy(CY[IND_REF[:, tau]]), 
                                            np.copy(COEFF_REF[IND_REF[:, tau], tau].flatten()))
            
            out.append([ydev1, cydev1, coeff1])
            out[tau][0], ind = sun.sort_SYT(out[tau][0], order='LLOS')
            out[tau][1] = out[tau][1][ind]
            out[tau][2] = out[tau][2][ind]
            
            # now the SYTs in ydev1 are not necessarily stored in ascending 
            # order of LLOS
            # sort SYTs according to Y (--> LLOS)
            #COEFF_FINAL[:, tau] = sun.sort_development(Y, ydev1, coeff1)
    
    #return Y, CY, COEFF_FINAL
    if nu1nu2nu==1:
        return out[0]
    
    return out

