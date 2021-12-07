from enum import Enum
from abc import ABC
import structlog as logging
import simpleaudio as sa
import soundfile as sf
import librosa
import numpy as np
import samplerate
from scipy import signal
import tensorflow as tf

from acte.settings import FRAMERATE, AUDIO_DTYPE, SAFE_FILE_LENGTH, YES_INPUT, NO_INPUT
from acte.util.files import get_audio_for_example, create_unique_appendix

logger = logging.getLogger(__name__)


class Strategy(ABC):
    def __init__(self, index):
        self.index = index

    @staticmethod
    def get_audio(example, start, duration):
        return get_audio_for_example(example, start, duration)

    @staticmethod
    def name():
        raise NotImplementedError


class NoStrategy(Strategy):
    @staticmethod
    def get_audio(example, start, duration):
        return Strategy.get_audio(example, start, duration)

    @staticmethod
    def name():
        return NoStrategy.__name__


class TimeStrategy(Strategy):
    def get_audio(self, example, start, duration):
        return librosa.util.fix_length(
            samplerate.resample(
                Strategy.get_audio(example, start, duration * self.rate + 0.01),
                1 / self.rate,
                "sinc_fastest",
            ),
            int(duration * FRAMERATE),
        )

    def name(self):
        return f"{TimeStrategy.__name__} {self.rate:1.2f}x"


class SpeedUp(TimeStrategy):
    def __init__(self, index):
        np.random.seed(index)
        self.rate = 1 + abs(np.random.normal(0, 0.1))
        super().__init__(index)


class SlowDown(TimeStrategy):
    def __init__(self, index):
        np.random.seed(index)
        self.rate = max(0.5, 1 - abs(np.random.normal(0, 0.1)))
        super().__init__(index)


class Attenuation(Strategy):
    def __init__(self, index):
        np.random.seed(index)
        self.multiplier = 1 - np.random.uniform(0, 0.99)
        super().__init__(index)

    def name(self):
        return f"Attenuate by {self.multiplier:1.2f}%"

    def get_audio(self, example, start, duration):
        return super().get_audio(example, start, duration) * self.multiplier


class HighPassFilter(Strategy):
    def __init__(self, index):
        np.random.seed(index)
        self.filter_pass_freq = np.random.uniform(200, 2000)

        super().__init__(index)

    def name(self):
        return f"High pass with cutoff {self.filter_pass_freq:1.2f} Hz"

    def get_audio(self, example, start, duration):
        filter_stop_freq = 20
        filter_order = 51

        # High-pass filter
        nyquist_rate = FRAMERATE / 2.0
        desired = (0, 0, 1, 1)
        bands = (0, filter_stop_freq, self.filter_pass_freq, nyquist_rate)
        filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

        # Apply high-pass filter
        return signal.filtfilt(
            filter_coefs, [1], super().get_audio(example, start, duration)
        ).astype(AUDIO_DTYPE)


class AugmentationStrategy(Enum):
    NO_STRATEGY = NoStrategy
    SPEED_UP = SpeedUp
    SLOW_DOWN = SlowDown
    ATTENUTATION = Attenuation
    HIGH_PASS_FILTER = HighPassFilter


def get_random_strategies(num_strategies, augmentation_strategies=None):
    if augmentation_strategies is None:
        augmentation_strategies = list([strategy for strategy in AugmentationStrategy])
    num_different_strategies = len(augmentation_strategies)
    return [
        augmentation_strategies[index % num_different_strategies].value(index)
        for index in range(num_strategies)
    ]


def get_audio_for_prediction(
    example, start=0, duration=SAFE_FILE_LENGTH, strategy=NoStrategy,
):
    return strategy.get_audio(example, start, duration)


def play_image(image, example_name, stft_chunk, hop_length, prompt_save=False):
    audio = transform_image_to_audio(image, stft_chunk, hop_length)
    return play_audio(audio, example_name, prompt_save=prompt_save)


def play_audio(audio, example_name, prompt_save=False):
    audio *= np.iinfo(np.int16).max
    # convert to 16-bit data
    audio = audio.astype(np.int16)
    logger.info(
        "Playing sample", name=example_name, seconds_length=audio.size / FRAMERATE
    )
    play_obj = sa.play_buffer(audio, 1, 2, FRAMERATE)
    play_obj.wait_done()
    logger.info("Done playing")

    if prompt_save:
        user_input = None
        while user_input not in {YES_INPUT, NO_INPUT}:
            user_input = input(f"Save audio? [{YES_INPUT}/{NO_INPUT}]")
        if user_input == YES_INPUT:
            destination_name = f"{example_name}{create_unique_appendix()}.wav"
            sf.write(destination_name, audio, FRAMERATE)
            logger.info("Saving audio", destination=destination_name)
        elif user_input == NO_INPUT:
            pass
        else:
            raise ValueError("Unknown input")


def transform_audio_to_image(audio, stft_chunk, hop_length):
    return tf.math.pow(
        tf.abs(
            tf.signal.stft(
                audio, stft_chunk, hop_length, window_fn=tf.signal.hann_window
            )
        ),
        0.25,
    )


def transform_image_to_audio(image, stft_chunk, hop_length):
    image = np.power(np.transpose(image), 4)
    audio = librosa.griffinlim(image, hop_length=hop_length, win_length=stft_chunk)
    # Remove the last bit of audio because it's garbage and causes pain
    return audio[:-stft_chunk]
