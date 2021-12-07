from collections import deque
import tensorflow as tf
import numpy as np
import structlog as logging

import matplotlib.pyplot as plt

from acte.settings import AUDIO_DTYPE, FRAMERATE
from acte.util.audio import transform_audio_to_image

logger = logging.getLogger(__name__)


A_FOURTH = 1 / 4
A_HALF = 1 / 2


class STFTVisualizer:
    colors = ["r", "y", "b", "g"]

    def visualize_audio(
        self, title_text, audio, *plottables, stft_chunk, hop_length,
    ):
        audio_batch = tf.reshape(audio, [1, -1])
        image = (
            transform_audio_to_image(audio_batch, stft_chunk, hop_length)
            .numpy()
            .astype(AUDIO_DTYPE)[0]
        )
        image = np.flip(np.rot90(image, k=3), axis=1)
        self.visualize_spectrogram(
            title_text, image, *plottables, hop_length=hop_length
        )

    def visualize_tmog_spectrogram(self, title_text, image, *plottables, hop_length):
        image = np.flip(np.rot90(image, k=3), axis=1)
        return self.visualize_spectrogram(
            title_text, image, *plottables, hop_length=hop_length,
        )

    def visualize_spectrogram(self, title_text, image, *plottables, hop_length):
        try:
            plt.clf()
            length = image.shape[1] * hop_length / FRAMERATE
            x_coords = (np.arange(image.shape[1] + 1) / image.shape[1]) * length
            y_coords = (np.arange(image.shape[0] + 1)) / image.shape[0]
            plt.pcolormesh(x_coords, y_coords, image)

            plt.xlim((0, length))
            plt.ylim((0, 1))

            num_plottables = len(plottables)
            if num_plottables:
                plottable_height = 1 / num_plottables
                for i, plottable in enumerate(plottables):
                    bottom = i * plottable_height
                    color = self.colors[i % len(self.colors)]
                    if isinstance(plottable, (np.ndarray, list)) and plottable.ndim > 1:
                        p_x_coords = (
                            np.arange(plottable.shape[1] + 1) / plottable.shape[1]
                        ) * length
                        p_y_coords = (
                            np.arange(plottable.shape[0] + 1)
                        ) / plottable.shape[0]
                        p_y_coords = p_y_coords * plottable_height + bottom
                        plottable = np.ma.masked_where(plottable < 0.5, plottable)
                        plt.pcolormesh(
                            p_x_coords,
                            p_y_coords,
                            plottable,
                            cmap="cool",
                            vmin=0.0,
                            vmax=1.0,
                        )
                    elif isinstance(plottable, (np.ndarray, list)):
                        for marker in plottable:
                            plt.axvline(
                                x=marker,
                                ymin=bottom,
                                ymax=bottom + plottable_height,
                                linewidth=3,
                                color=color,
                            )
                    elif isinstance(plottable, tuple):
                        plottable_x, plottable_y = plottable
                        if (  # This is the beat basset label case
                            isinstance(plottable_y, np.ndarray)
                            and plottable_y.shape[1] == 2
                        ):
                            labels_x = []
                            labels_y = []
                            for j, x in enumerate(plottable_x):
                                should_score = plottable_y[j, 0]
                                true_value = plottable_y[j, 1]
                                if should_score:
                                    labels_x.append(x)
                                    labels_y.append(true_value)
                            plt.plot(
                                np.array(labels_x),
                                np.array(labels_y) * plottable_height + bottom,
                                f"-{color}o",
                            )
                        else:
                            plt.plot(
                                plottable_x,
                                plottable_y * plottable_height + bottom,
                                f"-{color}o",
                            )

            logger.info(
                "Visualizing audio", title=title_text, image_shape=image.shape,
            )
            plt.title(title_text)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.exception("Failed to visualize", error=e)


class BeatBassetVisualizer:
    def __init__(self):
        self.length_of_marker = 40
        self.vertical_visual_history = 20
        self.marker_history = deque(maxlen=self.vertical_visual_history)

    def visualize_estimate(self, estimated_location, estimated_bpm):
        number_of_stars = int((1 - estimated_location) * self.length_of_marker)
        nice_string = (
            number_of_stars * "*" + (self.length_of_marker - number_of_stars) * " "
        )
        self.marker_history.append(nice_string)
        if len(self.marker_history) == self.vertical_visual_history:
            total_string = "|\n".join(self.marker_history)
            print(f"{total_string}| {estimated_bpm:.2f}")
