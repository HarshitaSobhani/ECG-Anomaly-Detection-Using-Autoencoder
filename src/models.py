"""Dense autoencoder architecture.

The model is trained to reconstruct its input (autoencoder(x) ~= x)
using only normal ECG sequences. See EXPLANATION.md for the theory behind
why this setup enables anomaly detection.
"""
import tensorflow as tf


def build_dense_autoencoder(n_timesteps: int) -> tf.keras.Model:
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_timesteps,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
    ], name="dense_encoder")

    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_timesteps, activation="sigmoid"),
    ], name="dense_decoder")

    model = tf.keras.Sequential([encoder, decoder], name="dense_autoencoder")
    model.compile(optimizer="adam", loss="mae")
    return model

