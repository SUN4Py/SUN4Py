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

import math

# Reference:
#   The Number of Partitions of the Integer N into M Nonzero Positive Integers
#   W. J. A. Colman
#   Mathematics of Computation, Vol. 39, Number 159, July 1982, Pages 213-224
#   https://www.ams.org/journals/mcom/1982-39-159/S0025-5718-1982-0658226-5/S0025-5718-1982-0658226-5.pdf

def p1(n):
    # number of partitions of n in 1 part
    return int(1)


def p2(n):
    # number of partitions of n in 2 parts
    p2 = n//2
    return int(p2)


def p3(n):
    # number of partitions of n in 3 parts
    p3 = (n**2+3)//(3*2*2)
    return int(p3)


def p4(n):
    # number of partitions of n in 4 parts
    p4 = (n**3 + 3*n**2 + (9*n*(-1)**n-9*n)/2 + 32)//(math.factorial(4)*math.factorial(3))
    return int(p4)


def pnm(n, m):
    # number of partitions of n in exactly m parts
    if m<=0:
        return int(0)
    if n<=0:
        return int(0)
    
    if n==m:
        return int(1)
    elif m>n:
        return int(0)
    
    if m==1:
        return p1(n)
    elif m==2:
        return p2(n)
    elif m==3:
        return p3(n)
    elif m==4:
        return p4(n)
    else:
        return pnm(n-m, m)+pnm(n-1, m-1)


def pstarnm(n, m):
    # number of partitions of n in at most m parts
    ps = int(0)
    for k in range(1, m+1):
        ps += pnm(n, k)
    return ps


def p(n):
    # number of partitions of n
    if n<=0:
        return int(0)
    pn = int(0)
    for k in range(1, n+1):
        pn += pnm(n, k)
    return pn
