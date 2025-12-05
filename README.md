# SUNPy

SUNPy (for SU(_N_)Py) is a Python library implementing the SU(_N_) symmetry, exploiting the Schur-Weyl duality between the SU(_N_) group and the symmetric group $S_N$. The library implements the symmetry through Standard Young Tableaux.

It allows solving the SU(_N_) Heisenberg model (or any other SU(_N_)-symmetric model) on a lattice making full use of the SU(_N_) symmetry.

## Features

- Compute basic quantities related to irreducible representations of SU(_N_) (dimension, number of standard Young tableaux, quadratic Casimir, etc ...).
- Generate standard Young tableaux for any irreducible representation of SU(_N_), with or without internal constraints.
- Solve Heisenberg-like models on any lattice, for any target irreducible representation, and any local constraints.
- Compute subduction coefficients.

By local constraints, we understand different SU(_N_) "spins", which are nothing but irreducible representations of the Lie algebra su(_N_).

SUNPy is by no means expected to execute on the largest systems. SUNPy has not been developped with the aim of reaching performance. More on this below.

## Prerequisites

SUNPy is a Python 3 implementation requiring:

- numpy
- scipy
- pytest


## Running tests

To run the tests, simply execute
```
pytest
```

## Running main file

An example main file is provided. Simply execute as

```
python3 main_ed.py
```

## Significant others

The big sister project of SUNPy is SUNHB, a performant C++ library for solving the SU(_N_) Heisenberg model (closed source).

For Clebsch-Gordan coefficients of SU(_N_), refer to Alex et al., https://arxiv.org/pdf/1009.0437

## License

The code is licensed under GNU GPL-v3.0 as given in the file LICENSE.

## Authors

Samuel Gozel

## References

Group Representation Theory for Physicists, Jin-Quan Chen, Jialung Ping & Fan Wang, World Scientific (2002)

Group Theory in Physics, Volume II, J. F. Cornwell, Academic Press (1984)

[Phys. Rev. Lett. 113, 127204 (2014)](https://doi.org/10.1103/PhysRevLett.113.127204)

[Phys. Rev. B 93, 155134 (2016)](https://doi.org/10.1103/PhysRevB.93.155134)

[Phys. Rev. B 96, 115159 (2017)](https://doi.org/10.1103/PhysRevB.96.115159)
