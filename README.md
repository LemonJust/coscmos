# coscmos

[![License BSD-3](https://img.shields.io/pypi/l/coscmos.svg?color=green)](https://github.com/LemonJust/coscmos/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/coscmos.svg?color=green)](https://pypi.org/project/coscmos)
[![Python Version](https://img.shields.io/pypi/pyversions/coscmos.svg?color=green)](https://python.org)
[![tests](https://github.com/LemonJust/coscmos/workflows/tests/badge.svg)](https://github.com/LemonJust/coscmos/actions)
[![codecov](https://codecov.io/gh/LemonJust/coscmos/branch/main/graph/badge.svg)](https://codecov.io/gh/LemonJust/coscmos)

**cosCMOS** ( "**co**rrect **sCMOS**", pronounced as cosmos ) is a Python package designed to cancel the fixed-pattern noise of sCMOS cameras.

The noise is estimated through a one-time calibration. The package contains methods to estimate the fixed-pattern noise component from the calibration data and remove it from the images.

----------------------------------
## Installation

You can install `coscmos` via [pip]:

    pip install coscmos



To install latest development version:

    pip install git+https://github.com/LemonJust/coscmos.git

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"coscmos" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause

[file an issue]: https://github.com/LemonJust/coscmos/issues

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
