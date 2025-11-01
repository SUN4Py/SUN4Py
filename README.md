# SUNPy
A Python3 library for solving SU(N) Heisenberg models

SUNPy (for SU(_N_)Py) is a Python library for solving the SU(_N_) Heisenberg model (or any other SU(_N_)-symmetric model) on a lattice making full use of the SU(_N_) symmetry through standard Young tableaux.

## Features

- Compute basic quantities related to irreducible representations of SU(_N_) (dimension, number of standard Young tableaux, quadratic Casimir, etc ...).
- Generate standard Young tableaux for any irreducible representation of SU(_N_).
- Solve Heisenberg-like models on any lattice, for any target irreducible representation, and any local constraints (even space-anisotropic).

By local constraints, we understand different SU(_N_) "spins" (or SU(_N_) qubits/qudits), which are nothing but irreducible representations of the Lie algebra su(_N_).

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
python3.10 main.py
```

## Significant others

The big sister project of SUNPy is SUNHB, a performant C++ library for solving the SU(_N_) Heisenberg model.

## License

The code is licensed under GNU GPL-v3.0 as given in the file LICENSE.

## Authors

Samuel Gozel

## References

[Phys. Rev. Lett. 113, 127204 (2014)](https://doi.org/10.1103/PhysRevLett.113.127204)

[Phys. Rev. B 93, 155134 (2016)](https://doi.org/10.1103/PhysRevB.93.155134)

[Phys. Rev. B 96, 115159 (2017)](https://doi.org/10.1103/PhysRevB.96.115159)
