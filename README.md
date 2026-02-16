# SU(_N_)4Py

SU(_N_)4Py is a Python library implementing the SU(_N_) symmetry, exploiting the Schur-Weyl duality between the SU(_N_) group and the symmetric group $S_n$. The library implements the symmetry through Standard Young Tableaux.

It allows solving the SU(_N_) Heisenberg model (or any other SU(_N_)-symmetric model) on a lattice making full use of the SU(_N_) symmetry.

## Features

- Compute basic quantities related to irreducible representations of SU(_N_) (dimension, number of standard Young tableaux, quadratic Casimir, etc ...).
- Generate standard Young tableaux for any irreducible representation of SU(_N_), with or without internal constraints.
- Solve Heisenberg-like models on any lattice, for any target irreducible representation and any local constraints, using Lanczos-based Exact Diagonalization.
- Compute subduction coefficients.
- Perform DMRG calculations.

By local constraints, we understand different SU(_N_) "spins", which are nothing but irreducible representations of the Lie algebra su(_N_).

SU(_N_)4Py is by no means expected to execute on the largest systems. SU(_N_)4Py has not been developped with the aim of reaching high performance (beyond the symmetry aspect). SU(_N_)4Py however provides all the algorithmic details, implements many tricks and will allow any user to easily port SU(_N_)4Py to their favorite high-performance frameworks, depending on their specific needs.

## Prerequisites

SU(_N_)4Py is a Python 3 implementation requiring:

- numpy
- scipy
- matplotlib

## Running tests

If you cloned the SUNPy repository, running tests might be a good way to start:
```
import pytest
pytest.main()
```

## Example codes

Several illustrative examples are provided in [SUNPy Examples](./examples) to guide the user through different use-cases of SUNPy.

## Significant others

SU(_N_)4Py relies on Standard Young Tableaux and subduction coefficients of SU(_N_). For Clebsch-Gordan coefficients of SU(_N_), refer to Alex et al., https://arxiv.org/pdf/1009.0437

## Contributions, bugs, improvements

Feel free to open an issue with details and to ping me. Contributions are welcome, if they follow the general spirit of SU(_N_)4Py and provide an improvement of the codebase (algorithms, performance, user-friendliness, etc ...)

## License

The code is licensed under GNU GPL-v3.0 as given in the file LICENSE.

## Citation

If you use SU(_N_)4Py in your work, you are welcome to cite it as explained in the `CITATION.cff` file.

Depending on your application, you are also invited to cite one/several of the references below.

## Author

Samuel Gozel

## References

The major references for the group theory aspects of SU(_N_)4Py are (here in chronological order):

- *On Quantitative Substitutional Analysis*, Alfred Young, Proc. London Math. Soc. s2-34, 196 (1932)
- *Substitutional Analysis*, D. E. Rutherford, Edinburgh University Press (1948)
- *Combinatorial Algorithms*, A. Nijenhuis & H. S. Wilf, Academic Press (1978)
- *Group Theory in Physics, Volume II*, J. F. Cornwell, Academic Press (1984)
- *Group Representation Theory for Physicists*, Jin-Quan Chen, Jialung Ping & Fan Wang, World Scientific (2002)

The algorithms provided in SU(_N_)4Py have been described in the following articles, some of which the author of SU(_N_)4Py is one of the main authors:

- [Phys. Rev. Lett. 113, 127204 (2014)](https://doi.org/10.1103/PhysRevLett.113.127204)
- [Phys. Rev. B 93, 155134 (2016)](https://doi.org/10.1103/PhysRevB.93.155134)
- [Phys. Rev. B 96, 115159 (2017)](https://doi.org/10.1103/PhysRevB.96.115159)
- [Phys. Rev. B 97, 134420 (2018)](https://doi.org/10.1103/PhysRevB.97.134420)
- [Phys. Rev. Lett. 125, 057202 (2020)](https://doi.org/10.1103/PhysRevLett.125.057202)
- [Phys. Rev. B 104, L180411 (2021)](https://doi.org/10.1103/PhysRevB.104.L180411)
