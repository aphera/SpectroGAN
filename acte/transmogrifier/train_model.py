import random

import structlog
from tensorflow import function, GradientTape
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import boto3

from acte.settings import EXAMPLE_DATA_BUCKET
from acte.transmogrifier.generator import prepare_generator
from acte.transmogrifier.model import (
    get_generator,
    generator_loss,
    discriminator_loss,
    get_discriminator,
)
from acte.transmogrifier.settings import (
    MODEL_NAME,
    EPOCHS,
    NEW_MODEL,
    TRAINING_BATCH_SIZE,
    LATENT_SIZE,
    generator_model_path,
    discriminator_model_path,
    TRAINING_STARTING_EPOCH,
    TRAINING_EXAMPLES_SIZE,
    LABEL_FLIP_CHANCE,
    IMAGE_WIDTH,
)

logger = structlog.getLogger(__name__)

s3 = boto3.resource("s3")


@function
def _train_step(
    images, generator, discriminator, generator_loss_metric, discriminator_loss_metric,
):
    noise = tf.random.normal([TRAINING_BATCH_SIZE, IMAGE_WIDTH, LATENT_SIZE])

    with GradientTape() as gen_tape, GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        if random.random() < LABEL_FLIP_CHANCE:
            # Every so often flip the labels
            disc_loss = discriminator_loss(fake_output, real_output)
        else:
            disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator.optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator.optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    generator_loss_metric(gen_loss)
    discriminator_loss_metric(disc_loss)


def train_tmog():
    starting_epoch = 0 if NEW_MODEL else TRAINING_STARTING_EPOCH

    generator = get_generator(batch_size=TRAINING_BATCH_SIZE)
    discriminator = get_discriminator(batch_size=TRAINING_BATCH_SIZE)

    num_examples_to_generate = TRAINING_EXAMPLES_SIZE
    example_seed = tf.random.normal(
        [num_examples_to_generate, IMAGE_WIDTH, LATENT_SIZE]
    )
    training_example_path = f"logs/{MODEL_NAME}/training_example.npy"

    logger.info("Training", new_model=NEW_MODEL, starting_epoch=starting_epoch)
    training_generator = prepare_generator()

    generator_writer = tf.summary.create_file_writer(
        f"logs/transmogrifier/{MODEL_NAME}/generator"
    )
    discriminator_writer = tf.summary.create_file_writer(
        f"logs/transmogrifier/{MODEL_NAME}/discriminator"
    )
    generator_loss_metric = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
    discriminator_loss_metric = tf.keras.metrics.Mean(
        "discriminator_loss", dtype=tf.float32
    )

    for epoch in range(starting_epoch, EPOCHS):

        for i in tqdm(range(training_generator.__len__())):
            image_batch = training_generator.__getitem__(i)
            _train_step(
                image_batch,
                generator,
                discriminator,
                generator_loss_metric,
                discriminator_loss_metric,
            )

        with generator_writer.as_default():
            tf.summary.scalar("loss", generator_loss_metric.result(), step=epoch)

        with discriminator_writer.as_default():
            tf.summary.scalar("loss", discriminator_loss_metric.result(), step=epoch)

        generator_loss_metric.reset_states()
        discriminator_loss_metric.reset_states()

        # Produce images for the GIF as you go
        predictions = generator(example_seed, training=False)
        np.save(training_example_path, predictions)
        s3.meta.client.upload_file(
            training_example_path,
            EXAMPLE_DATA_BUCKET,
            f"training_examples/{MODEL_NAME}/training_example_{epoch}.npy",
        )

        generator.save(generator_model_path())
        discriminator.save(discriminator_model_path())
        if (epoch + 1) % 5 == 0:
            s3.meta.client.upload_file(
                generator_model_path(),
                EXAMPLE_DATA_BUCKET,
                generator_model_path(epoch=epoch),
            )
            s3.meta.client.upload_file(
                discriminator_model_path(),
                EXAMPLE_DATA_BUCKET,
                discriminator_model_path(epoch=epoch),
            )

        logger.info("Completed epoch", epoch=epoch)

    logger.info("Done training", name=MODEL_NAME)


train_tmog()
