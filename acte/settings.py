import numpy as np
from tensorflow import random

from acte.util.logging import configure_logging

random.set_seed(13)

configure_logging()

FRAMERATE = 22050
AUDIO_DTYPE = np.float32

EXAMPLE_DATA_BUCKET = None
VALIDATION_SPLIT = 0.05

SAFE_FILE_LENGTH = 29.6
NONSAFE_FILE_LENGTH = 30.0


YES_INPUT = "Y"
NO_INPUT = "n"
