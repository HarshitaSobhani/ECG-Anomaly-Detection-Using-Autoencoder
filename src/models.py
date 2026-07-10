"""Dense and LSTM autoencoder architectures.

Both models are trained to reconstruct their input (autoencoder(x) ~= x)
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


def build_lstm_autoencoder(n_timesteps: int, latent_dim: int = 16) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_timesteps, 1)),
        # Encoder
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=True),
        tf.keras.layers.LSTM(latent_dim, activation="relu", return_sequences=False),
        # Bridge: repeat the latent vector once per output time step
        tf.keras.layers.RepeatVector(n_timesteps),
        # Decoder
        tf.keras.layers.LSTM(latent_dim, activation="relu", return_sequences=True),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid")),
    ], name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mae")
    return model
