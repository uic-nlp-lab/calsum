# Clinical ALignment SUMmarization

[![Python 3.9][python39-badge]][python39-link]
[![Python 3.10][python310-badge]][python310-link]

Summarize text using Component ALignment Abstract Meaning Representation
(CALAMR) alignment.  This is the source code for the paper [Abstract Meaning
Representation for Hospital Discharge Summarization].


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

## Citation

```bibtex
@misc{landesAbstractMeaningRepresentation2025,
  title = {Abstract {{Meaning Representation}} for {{Hospital Discharge Summarization}}},
  author = {Landes, Paul and Rao, Sitara and Chaise, Aaron Jeremy and Eugenio, Barbara Di},
  year = {2025},
  month = jun,
  number = {arXiv:2506.14101},
  eprint = {2506.14101},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2506.14101}
}
```


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2024 - 2025 Paul Landes


<!-- links -->
[python39-badge]: https://img.shields.io/badge/python-3.9-blue.svg
[python39-link]: https://www.python.org/downloads/release/python-390
[python310-badge]: https://img.shields.io/badge/python-3.10-blue.svg
[python310-link]: https://www.python.org/downloads/release/python-310

[Abstract Meaning Representation for Hospital Discharge Summarization]: https://arxiv.org/pdf/2506.14101
