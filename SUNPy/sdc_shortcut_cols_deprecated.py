# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np
import sys

import sun
import subduction
import subductionshortcut


def get_SDC_dmrg_develop_cols(N, nu, nu1, l1, nu2, l2, ref2firstLLOS=False, ref1firstLLOS=True):
    """
    Compute subduction coefficients in DMRG framework using the shortcut in Chen's
    method, developing irreps as products of their columns
    """
    
    print('CAUTION: This is deprecated function. Use subductionshortcut.get_SDC_dmrg')
    
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
    
    nu1nu2nu = sun.multiplicity_irrep_mixed(nu, np.array([nu1, nu2], dtype=int), N)
    assert(nu1nu2nu==1)
    # currently, we have not dealt with the case of multiplicities in this 
    # advanced method
    
    n = np.sum(nu)
    n1 = np.sum(nu1)
    n2 = np.sum(nu2)
    assert n==n1+n2
    
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
        
        sys.exit('Error: for a column-development method, the reference SYT must be the last one in LLOS order.')
        assert ref2firstLLOS==False
        # here we develop wrt columns, so we require the reference SYT to be the LAST
        # SYT in the increasing order of the LLOS
        
        # the reference is the first SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='LLOS')
    else:
        # the reference is the last SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='iLLOS')        
    
    nu2_col = sun.transpose_shape(nu2)
    nc_nu2 = nu2[0]
    
    # Construct all SYTs associated to the shape nu-nu1, in increasing order of LLOS
    Y = subductionshortcut.get_sSYT_constraint(nu, nu1, nu2, constraint='antisymm',
                            fill_nu1=True, fill_y1=y1)
    NY = Y.shape[0]
    assert NY>0
    
    if NY==1:
        coeffdev_r = np.array([1.0], dtype=float)
    else:
        
        Proj = np.zeros(shape=(NY, NY), dtype=float)
        
        for k in range(0, nc_nu2-1):
            
            # all numbers of the k-th column of y2_ref, numbered with global DMRG numbering
            col1 = np.flip( np.arange(n-1 - np.sum(nu2_col[0:k]), n-1 - np.sum(nu2_col[0:k+1]), -1) )
            lcol1 = len(col1)
            
            # all numbers of the (k+1)-th column of y2_ref, numbered with global DMRG numbering
            col2 = np.flip(np.arange(n-1 - np.sum(nu2_col[0:k+1]), n-1 - np.sum(nu2_col[0:k+2]), -1))
            
            # extract number situated at the top of the (k+1)-th column
            nkp1 = col2[-1] # last element because we have flipped col2
            assert(nkp1+1==col1[0])
            
            # all numbers of nu2 which are not in the k-th and (k+1)-th columns
            # of y2_ref (global numbering)
            ocols = np.setdiff1d(np.arange(n1, n), np.hstack((col1, col2)))
            
            Hk_diag = np.zeros(shape=(NY,1))
            Hk_offdiag = np.zeros(shape=(NY,NY))
            
            for j in range(0, NY):
                
                # find state which is antisymmetric wrt column k and column k+1
                #ydev_j, coeffdev_j = develop_antisymm_2columns(nu, Y[j], col1, col2)
                ydev_j, coeffdev_j = subductionshortcut.develop_symmetry_two_ensembles(nu, Y[j], col1, col2, symmetry='columns')
                Nydev_j = ydev_j.shape[0]
                
                # apply permutation which interchanges the last number of
                # column k with the first number of column k+1
                
                # diagonal part
                
                valdiag = 0.0
                
                for p in range(0, Nydev_j):
                
                    if (ydev_j[p, nkp1] == ydev_j[p, nkp1+1]):
                        
                        valdiag += coeffdev_j[p]**2
                        
                    else:
                        
                        cydev_j_p = sun.get_column(ydev_j[p])
                        
                        if (cydev_j_p[nkp1]==cydev_j_p[nkp1+1]):
                            valdiag -= coeffdev_j[p]**2
                        else:
                            ax = sun.get_axial_distance(ydev_j[p], cydev_j_p, nkp1, nkp1+1)
                            rho = - 1.0/ax
                            valdiag += rho*coeffdev_j[p]**2
                
                Hk_diag[j] = valdiag
                
                # Off-diagonal part
                
                if (j<NY-1):
                    
                    for i in range(j+1, NY):
                        
                        # check if all "other particles" (not in columns k and k+1)
                        # are situated in the same places in equivalence class i
                        # than in equivalence class j
                        condition1 = ( np.sum( abs( Y[i, ocols] - Y[j, ocols] ) ) == 0 )
                        
                        # extract rows of numbers of column k and column k+1 for
                        # each class
                        tempi = np.sort(Y[i, np.hstack((col1, col2))])
                        tempj = np.sort(Y[j, np.hstack((col1, col2))])
                        condition1prime = ( np.sum( abs(tempi-tempj) ) == 0 )
                        
                        # condition1prime is redundant
                        if condition1==True:
                            assert(condition1prime==True)
                        
                        condition1 *= condition1prime
                        
                        if condition1==True:
                            
                            ycol1_j = Y[j, col1]
                            ycol1_i = Y[i, col1]
                            C1 = np.intersect1d(ycol1_j, ycol1_i)
                            condition2_col1 = ( len(C1) == len(ycol1_j) - 1 )
                            
                            ycol2_j = Y[j, col2]
                            ycol2_i = Y[i, col2]
                            C2 = np.intersect1d(ycol2_j, ycol2_i)
                            condition2_col2 = ( len(C2) == len(ycol2_j)-1 )
                            
                            condition2 = condition2_col1 * condition2_col2
                            
                            if (condition2==True):
                                
                                r1 = np.setdiff1d(ycol1_j, C1).flatten()
                                r2 = np.setdiff1d(ycol1_i, C1).flatten()
                                r2r1 = np.array([r2[0], r1[0]], dtype=int)
                                
                                # find state of class i which is antisymmetric 
                                # wrt column k and column k+1
                                # ydev_i, coeffdev_i = develop_antisymm_2columns(nu, Y[i], col1, col2)
                                ydev_i, coeffdev_i = subductionshortcut.develop_symmetry_two_ensembles(nu, Y[i], col1, col2, symmetry='columns')
                                
                                ind_j = np.argwhere( np.sum(abs(ydev_j[:, nkp1:nkp1+2] - r2r1) , axis=1)==0 ).flatten()
                                ydev_j_p = ydev_j[ind_j, :]
                                coeffdev_j_p = coeffdev_j[ind_j]
                                
                                ind_i = np.argwhere( np.sum( abs(ydev_i[:, nkp1:nkp1+2] - np.flip(r2r1)), axis=1)==0 ).flatten()
                                ydev_i_p = ydev_i[ind_i, :]
                                coeffdev_i_p = coeffdev_i[ind_i]
                                
                                ydev_j_p[:, nkp1] = r1
                                ydev_j_p[:, nkp1+1] = r2
                                
                                cydev_j_p_0 = sun.get_column(ydev_j_p[0, :])
                                rho = 1.0/sun.get_axial_distance(ydev_j_p[0, :], cydev_j_p_0, nkp1+1, nkp1)
                                coeffdev_j_p = np.sqrt( 1.0 - rho**2 ) * coeffdev_j_p
                                
                                valij = sun.overlap(ydev_i_p, coeffdev_i_p, 
                                                    ydev_j_p, coeffdev_j_p, 
                                                    need_fullsimplify=False)
                                
                                Hk_offdiag[i, j] = valij
                        
            Projk = np.diagflat( Hk_diag ) + Hk_offdiag + Hk_offdiag.transpose()
            Projk -= np.diagflat( np.full(shape=(NY,), fill_value=1.0/lcol1, dtype=float) )
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
        
        for k in range(nc_nu2-1, -1, -1):
            
            # antisymmetrize wrt column k
            st = n1 + np.sum(nu2_col[k+1:])
            ed = n1 + np.sum(nu2_col[k:])            
            col1 = np.arange(st, ed)
            
            y_temp, coeff_temp = subductionshortcut.develop_symmetry_one_ensemble(nu, Y[t], col1, symmetry='column')
            N_temp = len(coeff_temp)
            
            ydev_t = np.matlib.repmat(ydev_t, N_temp, 1)
            y_temp = np.repeat(y_temp[:, col1], repeats=Nt, axis=0)
            ydev_t[:, col1] = y_temp
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
    coeffdev_ref, ind_ref = subduction.set_overall_phase(
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
    
    #  Deal with Yamanouchi relative phase
    
    if (len(ifd)==0):
        ydev_final = ydev_ref
        cydev_final = cydev_ref
        coeffdev_final = coeffdev_ref.flatten()
        '''
        ydev_final_new2 = np.copy(ydev_ref)
        cydev_final_new2 = np.copy(cydev_ref)
        coeffdev_final_new2 = np.copy(coeffdev_ref.flatten())
        
        ydev_final_new3 = np.copy(ydev_ref)
        cydev_final_new3 = np.copy(cydev_ref)
        coeffdev_final_new3 = np.copy(coeffdev_ref.flatten())
        '''
    else:
        
        # TO DO implement a better solution by grouping SYTs in ydev which have
        # equivalent positions and performing calculation on only 1 representative
        # in each group
        
        # find sequence of transpositions which brings y2_ref to y2
        sigma, rho = subduction.SYT_to_target(y2_ref, y2)
        
        # go from local to global numbering
        sigma = n - sigma - 2
        
        #::::::::::::::::::::::::::::::::::
        # New code
        #::::::::::::::::::::::::::::::::::
        
        # other technique to figure out the sequence of transpositions
        ifd = ifd[0]
        Nop = n2 - ifd - 1
        assert(Nop==len(sigma))
        sigmap = n - np.arange(ifd, n2-1) - 2
        assert(np.linalg.norm(sigmap==sigma))
        
        #::::::::::::::::::::::::::::::::::
        # End New code
        #::::::::::::::::::::::::::::::::::
        
        
        #::::::::::::::::::::::::::::::::::
        # BASE CODE
        #::::::::::::::::::::::::::::::::::
        '''
        ydev_ref_copy = np.copy(ydev_ref[ind_ref[:, 0]])
        cydev_ref_copy = np.copy(cydev_ref[ind_ref[:, 0]])
        coeffdev_ref_copy = np.copy(coeffdev_ref[ind_ref[:, 0]]).flatten()
        
        ts1 = time.time()
        ydev_final, cydev_final, coeffdev_final = subduction.apply_transpositions(
                                        nu, 
                                        sigma, 
                                        rho, 
                                        ydev_ref_copy, 
                                        cydev_ref_copy, 
                                        coeffdev_ref_copy)
        te1 = time.time()
        telapsed1 = te1 - ts1
        '''
        #::::::::::::::::::::::::::::::::::
        # END BASE CODE
        #::::::::::::::::::::::::::::::::::
        
        
        #::::::::::::::::::::::::::::::::::
        # NEW CODE new method
        #::::::::::::::::::::::::::::::::::
        
        ydev_ref_copy2 = np.copy(ydev_ref[ind_ref[:, 0]])
        cydev_ref_copy2 = np.copy(cydev_ref[ind_ref[:, 0]])
        coeffdev_ref_copy2 = np.copy(coeffdev_ref[ind_ref[:, 0]]).flatten()
        
        #ts2 = time.time()
        ydev_final, cydev_final, coeffdev_final = subduction.apply_transpositions_v2(
                                    nu, 
                                    sigma, 
                                    rho, 
                                    ydev_ref_copy2, 
                                    cydev_ref_copy2, 
                                    coeffdev_ref_copy2)
        #te2 = time.time()
        #telapsed2 = te2 - ts2
        
        #::::::::::::::::::::::::::::::::::
        # END NEW CODE basic new method
        #::::::::::::::::::::::::::::::::::
        
        #::::::::::::::::::::::::::::::::::
        # NEW CODE grouping new method
        #::::::::::::::::::::::::::::::::::
        
        # there are ifd particles which are correctly placed
        # these are the ifd last ones in the global DMRG numbering
        '''
        ydev_ref_copy3 = np.copy(ydev_ref[ind_ref[:, 0]])
        cydev_ref_copy3 = np.copy(cydev_ref[ind_ref[:, 0]])
        coeffdev_ref_copy3 = np.copy(coeffdev_ref[ind_ref[:, 0]]).flatten()
        
        ts3 = time.time()
        ydev_final_new3, cydev_final_new3, coeffdev_final_new3 = subduction.apply_transpositions_v3(
                                    nu, 
                                    sigma, 
                                    rho, 
                                    ydev_ref_copy3, 
                                    cydev_ref_copy3, 
                                    coeffdev_ref_copy3,
                                    ifd)
        te3 = time.time()
        telapsed3 = te3 - ts3
        '''
        #::::::::::::::::::::::::::::::::::
        # END NEW CODE grouping new method
        #::::::::::::::::::::::::::::::::::
        
    # sort SYTs in ascending order of LLOS
    ydev_final, ind = sun.sort_SYT(ydev_final, order='LLOS')
    cydev_final = cydev_final[ind, :]
    coeffdev_final = coeffdev_final[ind]
    
    '''
    ydev_final_new2, ind2 = sun.sort_SYT(ydev_final_new2, order='LLOS')
    cydev_final_new2 = cydev_final_new2[ind2, :]
    coeffdev_final_new2 = coeffdev_final_new2[ind2]
    
    ydev_final_new3, ind3 = sun.sort_SYT(ydev_final_new3, order='LLOS')
    cydev_final_new3 = cydev_final_new3[ind3, :]
    coeffdev_final_new3 = coeffdev_final_new3[ind3]
    '''
    
    #::::::::::::::::::::::::::::::::::
    # CHECK
    #::::::::::::::::::::::::::::::::::
    '''
    assert(ydev_final_new2.shape==ydev_final.shape)
    assert(ydev_final_new3.shape==ydev_final.shape)
    
    assert(np.linalg.norm(ydev_final_new2 - ydev_final)==0)
    assert(np.linalg.norm(ydev_final_new3 - ydev_final)==0)
    
    assert(np.linalg.norm(cydev_final_new2 - cydev_final)==0)
    assert(np.linalg.norm(cydev_final_new3 - cydev_final)==0)
    
    assert(np.linalg.norm(coeffdev_final_new2 - coeffdev_final)<1.0e-12)
    assert(np.linalg.norm(coeffdev_final_new3 - coeffdev_final)<1.0e-12)
    '''
    
    return ydev_final, cydev_final, coeffdev_final
