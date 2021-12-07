import structlog
import numpy as np

from acte.transmogrifier.settings import (
    STFT_CHUNK,
    HOP_LENGTH,
    HOP_SECONDS,
    IMAGE_WIDTH,
)
from acte.util.audio import (
    get_audio_for_prediction,
    NoStrategy,
    transform_audio_to_image,
)

logger = structlog.getLogger(__name__)


def get_image_width_seconds_conservative():
    return IMAGE_WIDTH * HOP_SECONDS + 0.1


CLIP = 3.0


def get_image_seed_and_label_for_tmog(example, image_start_time, strategy=NoStrategy):
    image = transform_audio_to_image(
        get_audio_for_prediction(
            example,
            image_start_time,
            get_image_width_seconds_conservative(),
            strategy=strategy,
        ),
        stft_chunk=STFT_CHUNK,
        hop_length=HOP_LENGTH,
    )
    image = np.clip(image, 0, CLIP)
    image = np.divide(image, CLIP)
    image = np.subtract(np.multiply(image, 2.0), 1.0)
    image = np.expand_dims(image, axis=-1)
    return image[:IMAGE_WIDTH, :, :]


def rescale_output(output):
    output = np.divide(np.add(output, 1.0), 2.0)
    output = np.multiply(output, CLIP)
    return output.squeeze()
