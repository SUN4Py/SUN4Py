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

from sun4py.dmrg.rme.rmefund import RMEEngineFund

# In this example, we show how to compute the list of Reduced Matrix Elements (RMEs)
# of the Heisenberg interaction using subduction coefficients, for a use in 
# Density Matrix Renormalization Group (DMRG) calculations.

N = int(3) # SU(N)
num_irreps = int(16)
# num_irreps defines the number of irreps to consider, where irreps are ordered
# in ascending order of their quadratic Casimir eigenvalue

# light-initialie a RME calculator
rme_engine = RMEEngineFund(N, 
                           num_irreps, 
                           target='GS', 
                           tech='shortcut_cols', 
                           restarting=True, 
                           checkpointing=False, 
                           chkpt_method='log2')
# Some arguments may be particularly useful for long calculations:
# - restarting=True : search a previously computed list of RMEs, and use it to
#                     restart the calculation of a new list (with either more 
#                     or less irreps in the new list)
# - checkpointing=True : checkpoint the calculation of RMEs, useful to prevent
#                        the loss of calculated RMEs in, eg, a job with strict
#                        wall-time
# - chkpt_method: 'log2' (recommended) or 'custom' : checkpointing strategy
#                 for 'custom', the user should provide the values of num_irreps
#                 that need to be checkpointed, with the argument 
#                 chkpt_ni=np.array([...], dtype=int)
# 
# Launch the calculation
rme_engine.run()
