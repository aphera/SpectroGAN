import numpy as np
from acte.transmogrifier.data_utils import rescale_output
from acte.transmogrifier.settings import (
    HOP_LENGTH,
    STFT_CHUNK,
    MODEL_NAME,
    TRAINING_EXAMPLES_SIZE,
)
from acte.util.audio import play_image
from acte.util.visualize_data import STFTVisualizer


def visualize_training_example():
    training_example = "training_example"
    examples = np.load(f"logs/{MODEL_NAME}/{training_example}.npy")
    for i in range(TRAINING_EXAMPLES_SIZE):
        image = rescale_output(examples[i])
        visualizer = STFTVisualizer()
        name = f"{training_example} {i}"
        visualizer.visualize_tmog_spectrogram(name, image, hop_length=HOP_LENGTH)
        play_image(image, name, STFT_CHUNK, HOP_LENGTH, prompt_save=False)
