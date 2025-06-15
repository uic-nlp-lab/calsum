# Clinical ALignment SUMmarization

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.9][python39-badge]][python39-link]
[![Python 3.10][python310-badge]][python310-link]
[![Build Status][build-badge]][build-link]

Summarize text using Component ALignment Abstract Meaning Representation
(CALAMR) alignment.


## Documentation

See the [full documentation](https://plandes.github.io/calsum/index.html).
The [API reference](https://plandes.github.io/calsum/api.html) is also
available.


## Reproducing Results

1. Install Python 3.11
1. Install dependencies: `pip install -r src/python/requirements-all.txt`
1. Align admissions: `./calsum align`
1. Remove alignment error files: `./calsum rmalignerrs`
1. Match sentences: `./calsum match`
1. Remove match error files: `./calsum rmmatcherrs`
1. Vectorize batches: `./calsum batch`
1. Train and test the model: `./calsum traintest`
1. Generate the admissions for the physicians' review: `./calsum generate`
1. Create tables used in the paper: `./calsum report`


## Sub Projects

* [Clincial Domain Abstract Meaning Representation Graphs](https://github.com/plandes/clinicamr)
* [ALignment SUMmarization](https://github.com/plandes/alsum)


## Obtaining

The easiest way to install the command line program is via the `pip` installer:
```bash
pip3 install zensols.calsum
```

Binaries are also available on [pypi].


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2024 - 2025 Paul Landes


<!-- links -->
[pypi]: https://pypi.org/project/zensols.calsum/
[pypi-link]: https://pypi.python.org/pypi/zensols.calsum
[pypi-badge]: https://img.shields.io/pypi/v/zensols.calsum.svg
[python39-badge]: https://img.shields.io/badge/python-3.9-blue.svg
[python39-link]: https://www.python.org/downloads/release/python-390
[python310-badge]: https://img.shields.io/badge/python-3.10-blue.svg
[python310-link]: https://www.python.org/downloads/release/python-310
[build-badge]: https://github.com/plandes/calsum/workflows/CI/badge.svg
[build-link]: https://github.com/plandes/calsum/actions
