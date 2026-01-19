# SUNPy

SUNPy (for SU(_N_)Py) is a Python library implementing the SU(_N_) symmetry, exploiting the Schur-Weyl duality between the SU(_N_) group and the symmetric group $S_N$. The library implements the symmetry through Standard Young Tableaux.

It allows solving the SU(_N_) Heisenberg model (or any other SU(_N_)-symmetric model) on a lattice making full use of the SU(_N_) symmetry.

## Features

- Compute basic quantities related to irreducible representations of SU(_N_) (dimension, number of standard Young tableaux, quadratic Casimir, etc ...).
- Generate standard Young tableaux for any irreducible representation of SU(_N_), with or without internal constraints.
- Solve Heisenberg-like models on any lattice, for any target irreducible representation, and any local constraints, using Exact Diagonalization.
- Compute subduction coefficients.
- Perform DMRG calculations.

By local constraints, we understand different SU(_N_) "spins", which are nothing but irreducible representations of the Lie algebra su(_N_).

SUNPy is by no means expected to execute on the largest systems. SUNPy has not been developped with the aim of reaching high performance (beyond the symmetry aspect). SUNPy however provides all the algorithmic details, implements many tricks and will allow any user to easily port SUNPy to their favorite high-performance frameworks, depending on their specific needs.

## Prerequisites

SUNPy is a Python 3 implementation requiring:

- numpy
- scipy
- pytest


## Running tests

Running tests might be a good way to start:
```
import pytest
pytest.main()
```

## Running main file

Three example main files are provided: `main_ed.py` shows how to perform ED calculations, `main_rme.py` how to compute the lists of reduced matrix elements of the interaction necessary for DMRG, and `main_dmrg.py` illustrates a simple DMRG calculation.

## Significant others

SUNPy relies on Standard Young Tableaux and subduction coefficients of SU(_N_). For Clebsch-Gordan coefficients of SU(_N_), refer to Alex et al., https://arxiv.org/pdf/1009.0437

## Contributions, bugs, improvements

Feel free to open an issue with details and to ping me. Contributions are welcome, if they follow the general spirit of SUNPy and provide an improvement of the codebase (algorithms, performance, user-friendliness, etc ...)

## License

The code is licensed under GNU GPL-v3.0 as given in the file LICENSE.

## Citation

If you use SUNPy in your work, you are welcome to cite it as explained in the `CITATION.cff` file.

If you feel generous, you are also welcome to cite any of the below references.

## Author

Samuel Gozel

## References

Two major references for the group theory aspects of SUNPy are (with an emphasis on the first one):

Group Representation Theory for Physicists, Jin-Quan Chen, Jialung Ping & Fan Wang, World Scientific (2002)

Group Theory in Physics, Volume II, J. F. Cornwell, Academic Press (1984)

The algorithms provided in SUNPy have been described in the following articles, some of which the author of SUNPy is one of the main authors:

[Phys. Rev. Lett. 113, 127204 (2014)](https://doi.org/10.1103/PhysRevLett.113.127204)

[Phys. Rev. B 93, 155134 (2016)](https://doi.org/10.1103/PhysRevB.93.155134)

[Phys. Rev. B 96, 115159 (2017)](https://doi.org/10.1103/PhysRevB.96.115159)

[Phys. Rev. B 97, 134420 (2018)](https://doi.org/10.1103/PhysRevB.97.134420)

[Phys. Rev. Lett. 125, 057202 (2020)](https://doi.org/10.1103/PhysRevLett.125.057202)

[Phys. Rev. B 104, L180411 (2021)](https://doi.org/10.1103/PhysRevB.104.L180411)
