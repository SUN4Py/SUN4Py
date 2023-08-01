# SUNPy
A Python3 library for solving SU(N) Heisenberg models

SUNPy (for SU(_N_)Py) is a Python library for solving the SU(_N_) Heisenberg model on a lattice making full use of the SU(_N_) symmetry through standard Young tableaux.

## Features

- Compute the number of partitions of an integer _n_.
- Compute basic quantities related to irreps of SU(_N_) (dimension, number of SYTs, quadratic Casimir, etc ...).
- Generate standard Young tableaux for any irreducible representation of SU(_N_).
- Solve Heisenberg-like models on any lattice, for any target irreducible representation.
- Currently only for the fundamental, symmetric or antisymmetric irrep on each site. Any irrep on each site still to be implemented.

SUNPy is by no means expected to execute on large systems. SUNPy has not been developped with the aim of reaching performance.

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

Samuel Gozel, [samuel.gozel@psi.ch](mailto:samuel.gozel@psi.ch)

## References

[Phys. Rev. Lett. 113, 127204 (2014)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.127204)

[Phys. Rev. B 93, 155134 (2016)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.155134)
