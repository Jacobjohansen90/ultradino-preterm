# Finetune Ultradino models

This package is providing a simple way to load pretrained UltraDINO (https://lab.compute.dtu.dk/sonai/dinus) models with minimal dependencies.

See [examples/load_pretrained_ultradino.py](examples/load_pretrained_ultradino.py) for an example of how to use the model

## Installation

Run the following while standing in the root directory (containing the
`pyproject.toml` file):

```bash
pip install .
```

For development run instead:

```bash
pip install -e .[dev]
```

## Run tests

First install for development (see above). Then run the following while standing
in the root directory:

```bash
pytest
```

NB: The tests rely on data files only found on the SONAI server.
