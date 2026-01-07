# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np

from sunpy.sun import sun
from sunpy.sdc.common import sdcutils
from sunpy.sdc.dmrg import utils



def get_SDC(N, nu, nu1, l1, nu2, l2, ref2firstLLOS=True, ref1firstLLOS=True):
    """
    Compute subduction coefficients in DMRG framework using the shortcut in Chen's
    method, developing irreps as products of their columns or rows
    
    Parameters
    ----------
    N : int
        SU(N)
    nu : numpy array
        global irrep with n particles
    nu1 : numpy array
        irrep with n1 particles
    l1 : int
        bottom corner position of last particle in nu1
    nu2 : numpy array
        irrep with n2 particles, n=n1+n2
    l2 : int
        bottom corner position of last particle in nu2
    ref2firstLLOS : bool [OPTIONAL][DEFAULT: True]
        if True, the first SYT in the ascending order of the LLOS is used as 
        reference for fixing the overall phase convention. This corresponds to
        a development accross rows
        if False, the last SYT is used (namely the first in the descending 
        order of LLOS). This corresponds to a development accross columns
    ref1firstLLOS : bool [OPTIONAL][DEFAULT: True]
        reference SYT used for y1
    
    Returns
    -------
    ydev_final : numpy array
        SYTs
    ydev_final : numpy array
        associated columns
    coeffdev_final : numpy array
        coefficients
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
    assert(nu1nu2nu==1)
    # currently, we have not dealt with the case of multiplicities in this 
    # advanced method
    
    # construct SYT for nu1
    nu1p = np.copy(nu1)
    nu1p[l1] -= 1
    if ref1firstLLOS==True:
        y1 = sun.index_to_SYT(0, alpha=nu1p, order='LLOS')
    else:
        y1 = sun.index_to_SYT(0, alpha=nu1p, order='iLLOS')
    y1 = np.hstack((y1, l1))
    
    # construct reference SYT for nu2
    
    if ref2firstLLOS==True:
        # we will develop wrt rows
        constraint = 'symm'
        symmetryshortcut = 'rows'
        si = 1.0
        nu2_rc = np.copy(nu2)
        nr_nu2 = len(np.argwhere(nu2>0).flatten()) # number of non-vanishing rows
        nb_rc_nu2 = nr_nu2
        # the reference is the first SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='LLOS')
    elif ref2firstLLOS==False:
        # we will develop wrt columns
        constraint = 'antisymm'
        symmetryshortcut = 'cols'
        si = -1.0
        nu2_col = sun.transpose_shape(nu2)
        nu2_rc = np.copy(nu2_col)
        nc_nu2 = nu2[0] # number of columns in nu2
        nb_rc_nu2 = nc_nu2
        # the reference is the last SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='iLLOS')        
    else:
        sys.exit('Problem: undefined ref2firstLLOS')
    
    # Construct all SYTs associated to the shape nu-nu1, in increasing order of LLOS
    Y = utils.get_sSYT_constraint(nu, nu1, nu2, constraint=constraint, 
                                  fill_nu1=True, 
                                  fill_y1=y1)
    NY = Y.shape[0]
    assert(NY>0)
    
    if symmetryshortcut=='rows':
        # need to compute the column positions
        CY = sun.get_column(Y)
    else:
        CY = Y
    
    if NY==1:
        coeffdev_r = np.array([1.0], dtype=float)
    else:
        
        Proj = np.zeros(shape=(NY, NY), dtype=float)
        
        for k in range(0, nb_rc_nu2-1):
            
            # all numbers of the k-th row/col of y2_ref, numbered with global DMRG numbering
            rc1 = np.flip( np.arange(n-1 - np.sum(nu2_rc[0:k]), n-1 - np.sum(nu2_rc[0:k+1]), -1) )
            lrc1 = len(rc1)
            
            # all numbers of the (k+1)-th row/col of y2_ref, numbered with global DMRG numbering
            rc2 = np.flip(np.arange(n-1 - np.sum(nu2_rc[0:k+1]), n-1 - np.sum(nu2_rc[0:k+2]), -1))
            
            # extract number situated in first position of the (k+1)-th row/col
            nkp1 = rc2[-1] # last element because we have flipped rc2
            assert(nkp1+1==rc1[0])
            
            # all numbers of nu2 which are not in the k-th and (k+1)-th rows/cols
            # of y2_ref (global numbering)
            orc = np.setdiff1d(np.arange(n1, n), np.hstack((rc1, rc2)))
            
            Hk_diag = np.zeros(shape=(NY,1))
            Hk_offdiag = np.zeros(shape=(NY,NY))
            
            for j in range(0, NY):
                
                # find state which is symmetric wrt row k and row k+1
                # respectively antisymmetric wrt col k and col k+1
                ydev_j, coeffdev_j = utils.develop_symmetry_two_ensembles(
                                        nu, Y[j], rc1, rc2, 
                                        symmetry=symmetryshortcut)
                cydev_j = sun.get_column(ydev_j)
                Nydev_j = ydev_j.shape[0]
                
                # apply permutation which interchanges the last number of
                # row/col k with the first number of row/col k+1
                
                # diagonal part
                
                valdiag = 0.0
                
                for p in range(0, Nydev_j):                
                    if (ydev_j[p, nkp1] == ydev_j[p, nkp1+1]):
                        valdiag += coeffdev_j[p]**2
                    else:
                        if (cydev_j[p, nkp1]==cydev_j[p, nkp1+1]):
                            valdiag -= coeffdev_j[p]**2
                        else:
                            ax = sun.get_axial_distance(ydev_j[p], cydev_j[p], nkp1, nkp1+1)
                            rho = - 1.0/ax
                            valdiag += rho*coeffdev_j[p]**2
                
                Hk_diag[j] = valdiag
                
                # Off-diagonal part
                
                if (j<NY-1):
                    
                    for i in range(j+1, NY):
                        
                        # check if all "other particles" (not in rows/cols k 
                        # and k+1) are situated in the same places in equivalence 
                        # class i than in equivalence class j
                        condition1 = ( np.sum( abs( Y[i, orc] - Y[j, orc] ) ) == 0 )
                        
                        if condition1==True:
                            
                            # recall that for column development, CY is just 
                            # and alias to Y
                            cyrow1_j = CY[j, rc1]
                            cyrow1_i = CY[i, rc1]
                            C1 = np.intersect1d(cyrow1_j, cyrow1_i)
                            condition2_rc1 = ( len(C1) == len(cyrow1_j) - 1 )
                            
                            cyrow2_j = CY[j, rc2]
                            cyrow2_i = CY[i, rc2]
                            C2 = np.intersect1d(cyrow2_j, cyrow2_i)
                            condition2_rc2 = ( len(C2) == len(cyrow2_j)-1 )
                            
                            condition2 = condition2_rc1 * condition2_rc2
                            
                            if (condition2==True):
                                
                                p1 = np.setdiff1d(cyrow1_j, C1).flatten()
                                p2 = np.setdiff1d(cyrow1_i, C1).flatten()
                                assert(len(p1)==1)
                                assert(len(p2)==1)
                                p2p1 = np.array([p2[0], p1[0]], dtype=int)
                                
                                # find state of class i which is symmetric 
                                # wrt row k and row k+1
                                # respectively antisymmetric wrt col k and col k+1
                                ydev_i, coeffdev_i = utils.develop_symmetry_two_ensembles(
                                                        nu, Y[i], rc1, rc2, 
                                                        symmetry=symmetryshortcut)
                                if symmetryshortcut=='rows':
                                    cydev_i = sun.get_column(ydev_i)
                                
                                if symmetryshortcut=='rows':
                                    ind_j = np.argwhere(np.sum(abs(cydev_j[:, nkp1:nkp1+2] - p2p1), axis=1)==0).flatten()
                                else:
                                    ind_j = np.argwhere( np.sum(abs(ydev_j[:, nkp1:nkp1+2] - p2p1) , axis=1)==0 ).flatten()
                                
                                ydev_j_p = ydev_j[ind_j, :]
                                cydev_j_p = cydev_j[ind_j, :]
                                coeffdev_j_p = coeffdev_j[ind_j]
                                
                                if symmetryshortcut=='rows':
                                    ind_i = np.argwhere(np.sum(abs(cydev_i[:, nkp1:nkp1+2] - np.flip(p2p1)), axis=1)==0).flatten()
                                else:
                                    ind_i = np.argwhere( np.sum( abs(ydev_i[:, nkp1:nkp1+2] - np.flip(p2p1)), axis=1)==0 ).flatten()
                                
                                ydev_i_p = ydev_i[ind_i, :]
                                if symmetryshortcut=='rows':
                                    cydev_i_p = cydev_i[ind_i, :]
                                coeffdev_i_p = coeffdev_i[ind_i]
                                
                                if symmetryshortcut=='rows':
                                    cydev_j_p[:, nkp1] = p1
                                    cydev_j_p[:, nkp1+1] = p2
                                    # get row positions from column positions
                                    ydev_j_p_0 = sun.get_column(cydev_j_p[0]) # sun.get_column is an involution
                                else:
                                    ydev_j_p[:, nkp1] = p1
                                    ydev_j_p[:, nkp1+1] = p2
                                    cydev_j_p_0 = sun.get_column(ydev_j_p[0, :])
                                
                                if symmetryshortcut=='rows':
                                    rho = 1.0/sun.get_axial_distance(ydev_j_p_0, cydev_j_p[0, :], nkp1, nkp1+1)
                                else:
                                    rho = 1.0/sun.get_axial_distance(ydev_j_p[0, :], cydev_j_p_0, nkp1, nkp1+1)
                                
                                coeffdev_j_p = np.sqrt( 1.0 - rho**2 ) * coeffdev_j_p
                                
                                if symmetryshortcut=='rows':
                                    valij = sun.overlap(cydev_i_p, coeffdev_i_p, 
                                                        cydev_j_p, coeffdev_j_p, 
                                                        need_fullsimplify=False)
                                else:
                                    valij = sun.overlap(ydev_i_p, coeffdev_i_p, 
                                                        ydev_j_p, coeffdev_j_p, 
                                                        need_fullsimplify=False)
                                
                                Hk_offdiag[i, j] = valij
                        
            Projk = np.diagflat( Hk_diag ) + Hk_offdiag + Hk_offdiag.transpose()
            Projk += si*np.diagflat(np.full(shape=(NY,), fill_value=1/lrc1, dtype=float))
            Proj += Projk @ Projk
            
        # Compute kernel of operator
        D, V = np.linalg.eigh(Proj)
        ind = np.argwhere(abs(D)<1e-10).flatten()
        if len(ind)==0:
            sys.exit('Problem: the kernel is empty.')
        V = V[:, ind]
        coeffdev_r = V.flatten()
    
    # Expand each class
    ydev_ref = np.copy(Y[0])
    cydev_ref = sun.get_column(ydev_ref)
    coeffdev_ref = np.array([0.0], dtype=float)
    
    for t in range(0, NY):
        
        ydev_t = Y[t]
        coeffdev_t = coeffdev_r[t]
        Nt = int(1)
        
        for k in range(nb_rc_nu2-1, -1, -1):
            
            # symmetrize wrt row/col k
            st = n1 + np.sum(nu2_rc[k+1:])
            ed = n1 + np.sum(nu2_rc[k:])
            rc1 = np.arange(st, ed)
            
            y_temp, coeff_temp = utils.develop_symmetry_one_ensemble(
                                    nu, Y[t], rc1, 
                                    symmetry=symmetryshortcut)
            N_temp = len(coeff_temp)
            
            ydev_t = np.matlib.repmat(ydev_t, N_temp, 1)
            y_temp = np.repeat(y_temp[:, rc1], repeats=Nt, axis=0)
            ydev_t[:, rc1] = y_temp
            coeffdev_t = np.multiply( np.matlib.repmat(coeffdev_t, 1, N_temp).flatten(), 
                                      np.repeat(coeff_temp, Nt) )
            
            Nt *= N_temp
        
        # TO DO need to define a sum_develop without column positions
        cydev_t = sun.get_column(ydev_t)
        ydev_ref, cydev_ref, coeffdev_ref = sun.sum_develop(ydev_ref, cydev_ref, coeffdev_ref, 
                                                            ydev_t, cydev_t, coeffdev_t)
    
    # order SYTs in ascending order of LLOS
    ydev_ref, ind = sun.sort_SYT(ydev_ref, order='LLOS')
    coeffdev_ref = coeffdev_ref[ind]
    coeffdev_ref[np.argwhere(abs(coeffdev_ref)<1.0e-12).flatten()] = 0.0
    
    # Deal with global phase
    coeffdev_ref, ind_ref = sdcutils.set_overall_phase(
                                np.reshape(coeffdev_ref, (len(coeffdev_ref), 1)))
    
    cydev_ref = sun.get_column(ydev_ref)
    
    # generate y2
    nu2p = np.copy(nu2)
    nu2p[l2] -= 1
    if ref2firstLLOS==True:
        y2 = sun.index_to_SYT(0, alpha=nu2p, order='LLOS')
    else:
        y2 = sun.index_to_SYT(0, alpha=nu2p, order='iLLOS')
    y2 = np.hstack((y2, l2))
    
    ifd = np.argwhere(abs(y2_ref-y2)>0).flatten()
    
    if (len(ifd)==0):
        ydev_final = ydev_ref
        cydev_final = cydev_ref
        coeffdev_final = coeffdev_ref.flatten()
    else:
        
        # find sequence of transpositions which brings y2_ref to y2
        sigma, rho = sun.SYT_to_target(y2_ref, y2)
        
        # go from local to global numbering
        sigma = n - sigma - 2
        
        # double-check sequence of transpositions
        ifd = ifd[0]
        Nop = n2 - ifd - 1
        assert(Nop==len(sigma))
        sigmap = n - np.arange(ifd, n2-1) - 2
        assert(np.linalg.norm(sigmap==sigma))
        
        ydev_ref_copy = np.copy(ydev_ref[ind_ref[:, 0]])
        cydev_ref_copy = np.copy(cydev_ref[ind_ref[:, 0]])
        coeffdev_ref_copy = np.copy(coeffdev_ref[ind_ref[:, 0]]).flatten()
        
        '''
        # base code - slow
        ydev_final, cydev_final, coeffdev_final = subduction.apply_transpositions(
                                        nu, 
                                        sigma, 
                                        rho, 
                                        ydev_ref_copy, 
                                        cydev_ref_copy, 
                                        coeffdev_ref_copy)
        '''
        
        ydev_final, cydev_final, coeffdev_final = sun.apply_transpositions_v2(
                                    nu, 
                                    sigma, 
                                    rho, 
                                    ydev_ref_copy, 
                                    cydev_ref_copy, 
                                    coeffdev_ref_copy)
        
    # sort SYTs in ascending order of LLOS
    ydev_final, ind = sun.sort_SYT(ydev_final, order='LLOS')
    cydev_final = cydev_final[ind, :]
    coeffdev_final = coeffdev_final[ind]
        
    return ydev_final, cydev_final, coeffdev_final
