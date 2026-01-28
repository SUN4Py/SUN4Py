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

from sunpy.dmrg.rme import rmefund
#from sunpy.dmrg.rme import rmesymm


N = int(3)
num_irreps = int(16)

rme_engine = rmefund.RMEEngineFund(N, num_irreps, 
                                   target='GS', 
                                   tech='shortcut_cols', 
                                   restarting=True, 
                                   checkpointing=False, 
                                   chkpt_method='log2')

rme_engine.run()


########################################
# SYMMETRIC IRREPS
########################################
'''
N = int(3)
m = int(3)
num_irreps = int(6)

rme_engineSymm = rmesymm.RMEEngineSymm(N, m, num_irreps, 
                                   #target='ES300_idmrg_edgeAdjoint',
                                   target='ES330_idmrg_edgeAdjoint',
                                   tech='shortcut_rows', 
                                   restarting=False, 
                                   checkpointing=False)

rme_engineSymm.run()
'''