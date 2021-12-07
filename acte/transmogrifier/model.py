import structlog
from tensorflow.python.keras.models import Model, load_model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

from acte.transmogrifier.settings import (
    IMAGE_HEIGHT,
    MODEL_NAME,
    NEW_MODEL,
    IMAGE_WIDTH,
    HOP_SECONDS,
    LATENT_SIZE,
    generator_model_path,
    discriminator_model_path,
)

logger = structlog.getLogger(__name__)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def _conv2d_transpose(input_layer, dimension, filters, kernel_size, dilation):
    if dimension == "time":
        kernel = (kernel_size, 1)
    elif dimension == "frequency":
        kernel = (1, kernel_size)
    else:
        raise ValueError("dimension must be either time or frequency")
    conv2d_transpose = layers.Conv2DTranspose(
        filters,
        kernel,
        padding="same",
        dilation_rate=(dilation, dilation),
        use_bias=False,
    )(input_layer)
    batch_norm = layers.BatchNormalization()(conv2d_transpose)
    return layers.LeakyReLU()(batch_norm)


def _conv1d(input_layer, filters, kernel_size, dilation_rate):
    conv1d = layers.Conv1D(
        filters, kernel_size, padding="same", dilation_rate=dilation_rate,
    )(input_layer)
    batch_norm = layers.BatchNormalization()(conv1d)
    leaky_relu = layers.LeakyReLU()(batch_norm)
    return layers.SpatialDropout1D(0.1)(leaky_relu)


def _conv2d(input_layer, filters, kernel_size, dilation):
    conv1d = layers.Conv2D(
        filters, (1, kernel_size), padding="same", dilation_rate=(1, dilation),
    )(input_layer)
    batch_norm = layers.BatchNormalization()(conv1d)
    leaky_relu = layers.LeakyReLU()(batch_norm)
    return layers.SpatialDropout2D(0.1)(leaky_relu)


def _create_new_model(batch_size, print_summary=True):
    inputs = layers.Input(
        (IMAGE_WIDTH, LATENT_SIZE,), name="input", batch_size=batch_size
    )

    dense_dim = 16
    beat_finder_output = layers.TimeDistributed(
        layers.Dense(IMAGE_HEIGHT * dense_dim, use_bias=False, activation="relu")
    )(inputs)

    reshaped_dense = layers.Reshape((IMAGE_WIDTH, IMAGE_HEIGHT, dense_dim))(
        beat_finder_output
    )

    f_filters = 128
    f = _conv2d_transpose(reshaped_dense, "frequency", f_filters, 3, dilation=1)
    f = _conv2d_transpose(f, "frequency", f_filters, 3, dilation=2)
    f = _conv2d_transpose(f, "frequency", f_filters, 3, dilation=4)
    f = _conv2d_transpose(f, "frequency", f_filters, 3, dilation=8)

    t_filters = 128
    t = _conv2d_transpose(reshaped_dense, "time", t_filters, 3, dilation=1)
    t = _conv2d_transpose(t, "time", t_filters, 3, dilation=2)
    t = _conv2d_transpose(t, "time", t_filters, 3, dilation=4)
    t = _conv2d_transpose(t, "time", t_filters, 3, dilation=8)

    # Combined output
    c = layers.Concatenate()([f, t])

    output = layers.Conv2DTranspose(
        1, (3, 3), padding="same", use_bias=False, activation="tanh",
    )(c)

    model = Model(inputs=inputs, outputs=output, name="generator")

    logger.info(
        "Built generator",
        batch_size=batch_size,
        image_width=IMAGE_WIDTH,
        image_width_seconds=IMAGE_WIDTH * HOP_SECONDS,
        name=MODEL_NAME,
    )

    if print_summary:
        model.summary(line_length=200)

    try:
        expected_output_shape = (batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1)
        model_output_shape = model.output_shape
        assert model_output_shape == expected_output_shape
    except AssertionError as error:
        logger.error(
            "Generator output shape does not match what we expect",
            expected_output_shape=expected_output_shape,
            model_output_shape=model_output_shape,
            error=error,
        )
        raise error
    model.compile(optimizer=Adam(1e-4))
    return model


def _create_new_discriminator(batch_size, print_summary=True):
    inputs = layers.Input(
        (IMAGE_WIDTH, IMAGE_HEIGHT, 1,), name="input", batch_size=batch_size
    )

    # Tone Tour
    f_filters = 16
    f = _conv2d(inputs, f_filters, 3, dilation=1)
    f = _conv2d(f, f_filters, 3, dilation=2)
    f = _conv2d(f, f_filters, 3, dilation=4)
    f = _conv2d(f, f_filters, 3, dilation=8)
    tone_tour_output = layers.Reshape((IMAGE_WIDTH, IMAGE_HEIGHT * f_filters))(f)

    # Beat Basset
    f_output = layers.Reshape((IMAGE_WIDTH, IMAGE_HEIGHT))(inputs)
    t_filters = 128
    t = _conv1d(f_output, t_filters, 3, dilation_rate=1)
    t = _conv1d(t, t_filters, 3, dilation_rate=2)
    t = _conv1d(t, t_filters, 3, dilation_rate=4)
    t = _conv1d(t, t_filters, 3, dilation_rate=8)
    beat_basset_output = _conv1d(t, t_filters, 3, dilation_rate=16)

    # Combined output
    c = layers.Concatenate()([tone_tour_output, beat_basset_output])
    output = layers.Dense(1)(layers.Flatten()(c))

    model = Model(inputs=inputs, outputs=output, name="discriminator")

    logger.info(
        "Built discriminator",
        batch_size=batch_size,
        image_width=IMAGE_WIDTH,
        image_width_seconds=IMAGE_WIDTH * HOP_SECONDS,
        name=MODEL_NAME,
    )

    if print_summary:
        model.summary(line_length=150)

    try:
        expected_output_shape = (batch_size, 1)
        model_output_shape = model.output_shape
        assert model_output_shape == expected_output_shape
    except AssertionError as error:
        logger.error(
            "Discriminator output shape does not match what we expect",
            expected_output_shape=expected_output_shape,
            model_output_shape=model_output_shape,
            error=error,
        )
        raise error
    model.compile(optimizer=Adam(1e-4))
    return model


def get_generator(batch_size, print_summary=True, epoch=None):
    if NEW_MODEL:
        model = _create_new_model(batch_size, print_summary)
    else:
        model = load_model(generator_model_path(epoch=epoch))
        logger.info("Loaded generator", name=MODEL_NAME)
    return model


def get_discriminator(batch_size, print_summary=True, epoch=None):
    if NEW_MODEL:
        model = _create_new_discriminator(batch_size, print_summary)
    else:
        model = load_model(discriminator_model_path(epoch=epoch))
        logger.info("Loaded discriminator", name=MODEL_NAME)
    return model
