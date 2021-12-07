import json
from datetime import datetime

import librosa
import numpy as np

import structlog

from acte.settings import FRAMERATE

logger = structlog.getLogger(__name__)


def create_unique_appendix():
    date_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    return f"_{date_str}"


def get_audio_for_example(example, start=0.0, duration=None):
    y, _ = librosa.load(
        example["audio_file"],
        sr=FRAMERATE,
        offset=start,
        duration=duration,
        res_type="sinc_fastest",
    )
    return y


# MAESTRO


def get_maestro_examples():
    with open("data/maestro/maestro-v3.0.0.json", "r") as file:
        data_file = json.loads(file.read())
    training_examples = []
    validation_examples = []
    for example_id, audio_file in data_file["audio_filename"].items():
        duration = data_file["duration"][example_id]
        midi_file = data_file["midi_filename"][example_id]
        example = {
            "id": example_id,
            "audio_file": f"data/maestro/maestro-v3.0.0/{audio_file}",
            "midi_file": f"data/maestro/maestro-v3.0.0/{midi_file}",
            "duration": duration,
        }
        if data_file["split"][example_id] == "train":
            training_examples.append(example)
        elif data_file["split"][example_id] == "validation":
            validation_examples.append(example)
    logger.info(
        "Created datasets",
        length_validation=len(validation_examples),
        length_training=len(training_examples),
    )
    return np.array(validation_examples), np.array(training_examples)
