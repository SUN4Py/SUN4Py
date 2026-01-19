# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np
import scipy.sparse

from sunpy.sun import sun



def set_overall_phase(COEFF_REF):
    """
    Set the overall phase of a series of SDCs
    
    Parameters
    ----------
    COEFF_REF : numpy array
        set of SDCs (stored in the columns)
    
    Returns
    -------
    COEFF_REF : numpy array
        rotated SDCs with correct phase convention
    IND_REF : list
        positions of non-zero elements for each sequence of SDCs in COEFF_REF
        IND_REF is a list of length COEFF_REF.shape[1]
        each element in IND_REF is a numpy array of indices for non-zero elements
        in output COEFF_REF
    """
    
    nu1nu2nu = COEFF_REF.shape[1]
    
    if (nu1nu2nu==1):
        # The coefficient of the first (in increasing order of the LLOS) non-zero
        # term must be positive
        ind_sdc = np.argwhere(abs(COEFF_REF[:, 0])>1.0e-12).flatten()
        if (COEFF_REF[ind_sdc[0], 0]<0):
            COEFF_REF *= -1.0
        IND_REF = [ind_sdc]
    else:
        if (nu1nu2nu>2):
            print('sdcutils.set_overall_phase(): multiplicity>2: abort. Code needs to perform the same operations recursively')
            sys.exit()
        
        ind_a = np.argwhere( abs(COEFF_REF[:, 0])<1.0e-12 ).flatten()
        ind_b = np.argwhere( abs(COEFF_REF[:, 1])<1.0e-12 ).flatten()
        
        if (len(ind_a)>0) | (len(ind_b)>0):
            sys.exit('This is very unlikely, but could cause a problem further down. If it occurs, that case needs to be treated.')
        
        # We apply a rotation in order to set to 0 the last component of one of
        # the vectors
        a = COEFF_REF[-1, 0]
        b = COEFF_REF[-1, 1]
        ii = COEFF_REF.shape[0] - 1
        
        while ( (abs(a**2+b**2)<1.0e-12) & (ii>=0) ): # necessary to avoid possible division by 0 in eta
            a = COEFF_REF[ii, 0]
            b = COEFF_REF[ii, 1]
            ii -= 1
        
        eta = np.sqrt( b**2/(a**2+b**2) )
        
        c1 = eta*COEFF_REF[:, 0] + np.sqrt(1.0-eta**2)*COEFF_REF[:, 1]
        c2 = np.sqrt(1.0-eta**2)*COEFF_REF[:, 0] - eta*COEFF_REF[:, 1]
        
        if abs(c1[-1])>1.0e-12:
            # we made the wrong choice of phase in lambdaa ==> need to correct
            c1 = eta*COEFF_REF[:, 0] - np.sqrt(1.0-eta**2)*COEFF_REF[:, 1]
            c2 = np.sqrt(1.0-eta**2)*COEFF_REF[:, 0] + eta*COEFF_REF[:, 1]
        
        ind_c1 = np.argwhere(abs(c1)>1.0e-12).flatten()
        if (c1[ind_c1[0]]<0):
            c1 *= -1
        ind_c2 = np.argwhere(abs(c2)>1.0e-12).flatten()
        if (c2[ind_c2[0]]<0):
            c2 *= -1
        
        COEFF_REF[:, 0] = c1
        COEFF_REF[:, 1] = c2
        
        IND_REF = [ind_c1, ind_c2]
    
    return COEFF_REF, IND_REF


def casimir_canonical_chain(nu, y):
    """
    Get the eigenvalues of the quadratic Casimir operator for the canonical 
    group chain of nu defined by SYT y.
    
    Parameters
    ----------
    nu : numpy array
        irrep
    y : numpy array
        SYT
    
    Returns
    -------
    casimir : numpy array
        eigenvalues of CSCO-II of y of nu
    
    References
    ----------
    [1]         Group Representation Theory for Physicists
                Jin-Quan Chen, Hialun Ping and Fan Wang
                World Scientific, 2nd edition, (2002)
    """
    
    n = np.sum(nu)
    nup = np.copy(nu)
    casimir = np.zeros(shape=(n,), dtype=float)
    for j in range(n-1, -1, -1):
        casimir[j] = sun.casimir_quadratic(nup)
        nup[y[j]] -= 1
    
    return casimir


def build_canonical_chain_projector(NY, n, n1, MatAdjaTranspo, casimir, ordering='dmrg'):
    """
    Build projection operator of the CSCO-II associated with input casimir values
    """
    
    M = scipy.sparse.csr_matrix((NY, NY))
    
    if ordering=='math':
        f = lambda j, k : (n1+k, n1+j)
    elif ordering=='dmrg':
        f = lambda j, k : (n-1-j, n-1-k)
    
    for j in range(1, n-n1):
        Mj = scipy.sparse.csr_matrix((NY, NY))    
        for k in range(0, j):
            transpo = np.array(f(j, k))
            sigma = sun.transposition_to_adjacent_transpositions(transpo)
            Mtemp = scipy.sparse.eye(NY)    
            for atr in sigma:
                Mtemp = Mtemp @ MatAdjaTranspo[atr-n1]
            Mj += Mtemp
        val = -(casimir[j] - casimir[j-1])
        Mj = Mj + val*scipy.sparse.eye(NY)
        M += (Mj @ Mj)
    
    return M
