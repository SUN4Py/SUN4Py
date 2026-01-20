# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np



def find_row(A, a):
    """
    Search array A for rows equal to a and return indices of matching rows
    """
    indices = np.argwhere(np.sum(abs(A - a), axis=1)==0).flatten()
    return indices


def find_col(A, a):
    """
    Search array A for columns equal to a and return indices of matching columns
    """
    indices = np.argwhere(np.sum(abs(A.transpose() - a), axis=1)==0).flatten()
    return indices


def intersect_row(A, B):
    """
    Find rows which appear in A AND in B
    
    Source - https://stackoverflow.com/a
    Posted by Joe Kington
    Retrieved 2025-11-16, License - CC BY-SA 3.0
    """
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}
    
    C, indA, indB = np.intersect1d(A.view(dtype), B.view(dtype), 
                                   assume_unique=True, return_indices=True)
    C = C.view(A.dtype).reshape(-1, ncols)
    
    return C, indA, indB


def sortrows(A):
    """
    Equivalent to Matlab's sortrows.
    
    Parameters
    ----------
    A : numpy array
    
    Returns
    -------
    B : numpy array
      sorted array of rows
    index :  numpy array 
        indices such that B == A[index, :]
    
    Remark
    ------
    See https://ch.mathworks.com/help/matlab/ref/double.sortrows.html
    """
    
    if A.shape[0]>1:
        tmp = []
        for i in range(A.shape[1]-1,-1,-1):
            tmp.append(tuple(A[:,i]))
        index = np.lexsort(tmp)
        B = np.copy(A)
        B = B[index,:]
    else:
        B = A
        index = np.array([0], dtype=int)
    
    return B, index


def reduce_by_divide(numvec, denomvec):
    """
    Reduce numvec (numerator) and denomvec (denominator) by finding common divisors
    
    Parameters
    ----------
    numvec : numpy array
        array of integer factors in numerator
    denomvec : numpy array
        array of integer factors in denominator
    
    Returns
    -------
    numvec : numpy array
        reduced array of integer factors in numerator
    denomvec : numpy array
        reduced array of integer factors in denominator
    
    Explanations
    ------------
    This function is useful when computing an integer fraction where both the
    numerator and the denominator are expressed as products of integers, provided
    in the input numvec and denomvec. Computing the ratio as np.prod(numvec)/np.prod(denomvec)
    is likely to overflow on 32bits (or even 64bits) integers.
    
    Example: Computing the number of SYTs for the irrep [5, 5, 5, 5, 5] of SU(5) (singlet irrep)
    
        multiplicity = 25!/(product of Hook lengths)
    
    where the sequence of Hook lengths is given by:
        [9, 8, 7, 6, 5, 8, 7, 6, 5, 4, 7, 6, 5, 4, 3, 6, 5, 4, 3, 2, 5, 4, 3, 2, 1] --> denomvec
    
    np.prod(np.arange(2, 26))/np.prod(denomvec) ---> garbage result, because of overflow
    
    After reduction, the mutiplicity is actually given by the product of the following factors:
        multiplicity = prod[5, 11, 11, 12, 13, 17, 19, 23] = 701149020
    """

    numvec = np.sort(numvec)
    denomvec = np.sort(denomvec)
    
    numvec = numvec[numvec>1]
    denomvec = denomvec[denomvec>1]
    
    if len(denomvec)==0:
        if len(numvec)==0:
            return np.array([1], dtype=int), np.array([1], dtype=int)
        else:
            return numvec, np.array([1], dtype=int)
    else:
        if len(numvec)==0:
            raise ValueError('numvec/denomvec does not represent an integer value.')
    
    # check for equality between factors in numerator and denominator
    for i in range(0, len(denomvec)):
        di = denomvec[i]
        ind = np.argwhere(numvec==di).flatten()
        if len(ind)>0:
            numvec[ind[0]] = 1
            denomvec[i] = 1
    
    # keep only non-1's
    numvec = numvec[numvec>1]
    denomvec = denomvec[denomvec>1]
    
    if len(denomvec)==0:
        if len(numvec)==0:
            return np.array([1], dtype=int), np.array([1], dtype=int)
        else:
            return numvec, np.array([1], dtype=int)
    else:
        if len(numvec)==0:
            raise ValueError('numvec/denomvec does not represent an integer value.')    
    
    # here, len(denomvec)>0
    
    # we further reduce by finding common divisors among prime numbers
    divisors = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], dtype=int)
    
    for j in range(0, len(denomvec)):
        # search decomposition of denomvec[j] as a product of prime numbers
        denomj = denomvec[j]
        denomj_decomposition = []
        while denomj>1:
            for d in divisors:
                if (denomj%d==0):
                    denomj_decomposition.append(d)
                    denomj = denomj // d
        denomj_decomposition = np.array(denomj_decomposition)
        denomj_decomposition = np.sort(denomj_decomposition)
        
        for d in denomj_decomposition:
            ind = np.argwhere(numvec%d==0).flatten()
            if len(ind)>0:
                numvec[ind[0]] = numvec[ind[0]] // d
                denomvec[j] = denomvec[j] // d
    
    numvec = numvec[numvec>1]
    denomvec = denomvec[denomvec>1]
    
    if len(numvec)==0:
        if len(denomvec)>0:
            raise ValueError('numvec/denomvec does not represent an integer value.')
        numvec = np.array([1], dtype=int)
    
    if len(denomvec)==0:
        denomvec = np.array([1], dtype=int)
    
    return numvec, denomvec
