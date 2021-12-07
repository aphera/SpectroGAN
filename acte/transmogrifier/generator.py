import numpy as np
import structlog

from acte.transmogrifier.data_utils import (
    get_image_seed_and_label_for_tmog,
    get_image_width_seconds_conservative,
)
from acte.transmogrifier.settings import TRAINING_BATCH_SIZE
from acte.util.audio import (
    get_random_strategies,
    AugmentationStrategy,
)
from acte.util.files import get_maestro_examples

logger = structlog.getLogger(__name__)


class SynthGenerator:
    def get_augmentation_strategy_example_and_image_start_time(
        self, index_in_epoch, index_in_batch
    ):
        index = index_in_epoch * TRAINING_BATCH_SIZE + index_in_batch
        file_index = self.file_indexes[index]

        augmentation_strategy = self.augmentation_strategies[index]

        example = self.examples[file_index]
        file_length = example["duration"]
        return (
            augmentation_strategy,
            example,
            np.random.random() * (file_length - get_image_width_seconds_conservative()),
        )

    def __init__(self, examples, shuffle=True):
        self.examples = examples
        self.num_examples = examples.size

        self.file_indexes = np.arange(self.num_examples)

        self.augmentation_strategies = get_random_strategies(
            self.num_examples,
            augmentation_strategies=[
                AugmentationStrategy.NO_STRATEGY,
                AugmentationStrategy.SLOW_DOWN,
                AugmentationStrategy.SPEED_UP,
            ],
        )

        self.shuffle = shuffle
        self._shuffle_indexes()

    def __len__(self):
        return int(self.num_examples / TRAINING_BATCH_SIZE)

    def __getitem__(self, index_in_epoch):
        strategies_examples_and_start_times = [
            self.get_augmentation_strategy_example_and_image_start_time(
                index_in_epoch, index_in_batch
            )
            for index_in_batch in range(TRAINING_BATCH_SIZE)
        ]

        return np.array(
            [
                get_image_seed_and_label_for_tmog(example, image_start_time, strategy)
                for strategy, example, image_start_time in strategies_examples_and_start_times
            ]
        )

    def _shuffle_indexes(self):
        if self.shuffle:
            logger.info("Shuffling indexes")
            np.random.shuffle(self.file_indexes)
            np.random.shuffle(self.augmentation_strategies)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self._shuffle_indexes()


class SimpleCaseGenerator:
    def __init__(self):
        self.length = 16

    def __len__(self):
        return self.length

    def __getitem__(self, index_in_epoch):
        image_seeds_and_labels = [
            get_image_seed_and_label_for_tmog(
                {"id": "test"}, 0, AugmentationStrategy.NO_STRATEGY.value(0)
            )
            for _ in range(TRAINING_BATCH_SIZE)
        ]
        return (
            np.array([seed for seed, _ in image_seeds_and_labels]),
            np.array([label for _, label in image_seeds_and_labels]),
        )


def prepare_generator():
    logger.info("Creating Generator")
    examples = np.concatenate(get_maestro_examples())
    return SynthGenerator(examples)
