# SU(N)4Py Examples

### [SU(N) Irreps](./01_Irreps)
This example illustrates how to deal with irreducible representations (irreps) of SU(_N_) in SUN4Py, and the calculation of some basic quantities related to these irreps (dimension, quadratic Casimir, ascendant and descendant shapes, etc ...).

### [SYTs](./02_SYTs)
This example illustrates how to generate Standard Young Tableaux (SYTs) associated to irreps of the permutation group $S_n$ (and thus, by the Schur-Weyl duality, of the irreps of SU(_N_) having $n$ particles and less or equal than $N$ rows).

### [ED fundamental](./03_ED_fund)
This example shows how to perform Exact Diagonalization (ED) calculations on systems with one particle per site at each site, namely the fundamental irrep of SU(_N_).

### [ED symmetric - application to the AKLT model (1/2)](./04_ED_symm_aklt)
This example illustrates the usefulness of SUN4Py to investigate the spectrum, and in particular the edge states for open boundary conditions, of a one-dimensional AKLT model having three-box symmetric irrep **10** of SU(3) at each site.

### [ED general - application to the AKLT model (2/2)](./05_ED_general_aklt)
This example shows how to use the generic Exact Diagonalization solver of SUN4Py to solve the AKLT model introduced in [ED symmetric - application to the AKLT model (1/2)](./04_ED_symmetric_aklt), when screening the edge states. 

### [Subduction coefficients of SU(_N_)](./06_SDCs)
This example shows how to compute subduction coefficients (SDCs) of SU(_N_).

### [DMRG Reduced matrix elements of the interaction](./07_DMRG_RME)
This example shows how to pre-compute the list of reduced matrix elements of the Heisenberg interaction necessary to perform DMRG calculations, relying on the calculation of subduction coefficients.

### [DMRG fundamental](./08_DMRG_fund)
This example shows how to perform Density Matrix Renormalization Group (DMRG) calculations on the SU(_N_) Heisenberg model with fundamental irrep at each site.
