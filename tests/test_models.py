import numpy as np

from src.models import build_dense_autoencoder, build_lstm_autoencoder

N_TIMESTEPS = 10


def test_dense_autoencoder_reconstructs_input_shape():
    model = build_dense_autoencoder(N_TIMESTEPS)
    x = np.random.rand(3, N_TIMESTEPS).astype("float32")
    out = model.predict(x, verbose=0)
    assert out.shape == x.shape


def test_lstm_autoencoder_reconstructs_input_shape():
    model = build_lstm_autoencoder(N_TIMESTEPS, latent_dim=4)
    x = np.random.rand(3, N_TIMESTEPS, 1).astype("float32")
    out = model.predict(x, verbose=0)
    assert out.shape == x.shape
