# SpectroGAN

To get started, install [Poetry](https://python-poetry.org/docs/#installation) then run `poetry install` in the project directory to install the dependencies.

`poetry run visualize-training-example` can be run to visualize and listen to samples from the model's output.

`python run acte/transmogrifier/train_model.py` can be run to train the model.

The script expects the [maestro-V3.0.0 dataset](https://magenta.tensorflow.org/datasets/maestro#v300) to be unzipped within a directory called `data/maestro`. Also, place the `maestro-v3.0.0.json` file within this directory.