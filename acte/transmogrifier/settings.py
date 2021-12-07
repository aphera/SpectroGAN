from acte.settings import FRAMERATE
from acte.util.files import create_unique_appendix

NEW_MODEL = True
TRAINING_STARTING_EPOCH = 0


def _model_path(model_type, unique_appendix, epoch):
    appendix = ""
    if unique_appendix:
        appendix = create_unique_appendix()
    if epoch:
        epoch = f"_{epoch}"
    else:
        epoch = ""
    return f"{MODEL_PREFIX}/{model_type}{epoch}{appendix}.h5"


def generator_model_path(unique_appendix=False, epoch=None):
    return _model_path("generator", unique_appendix, epoch)


def discriminator_model_path(unique_appendix=False, epoch=None):
    return _model_path("discriminator", unique_appendix, epoch)


MODEL_NAME = "23"
MODEL_PREFIX = f"models/transmogrifier/{MODEL_NAME}"

TRAINING_BATCH_SIZE = 16
EPOCHS = 2000
TRAINING_EXAMPLES_SIZE = 16
LABEL_FLIP_CHANCE = 0

STFT_CHUNK = 1024
HOP_LENGTH = 256
HOP_SECONDS = HOP_LENGTH / FRAMERATE
IMAGE_HEIGHT = 513

IMAGE_WIDTH = 256
LATENT_SIZE = 10
