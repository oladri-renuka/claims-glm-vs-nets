import numpy as np
import torch

from src.models.glm import PoissonGLM
from src.models.neural import NeuralFrequency, TabularAE

P = {"lr": 1e-3, "batch_size": 1000, "max_epochs": 2, "patience": 5, "hidden": [8, 4],
     "activation": "tanh", "embedding_dim": 2, "categorical_encoding": "embedding", "num_threads": 1,
     "latent_dim": 2, "anomaly_quantile": 0.99}


def test_glm_recovers_overall_rate_balance(policies):
    glm = PoissonGLM().fit(policies)
    # Canonical-link GLM with intercept satisfies the balance property on its training data.
    assert np.isclose(glm.predict(policies).sum(), policies.ClaimNb.sum(), rtol=1e-6)


def test_cann_starts_exactly_at_glm(policies):
    glm = PoissonGLM().fit(policies)
    torch.manual_seed(0)
    m = NeuralFrequency({**P, "max_epochs": 0}, glm=glm)
    m.fit_info = None
    from src.data.features import NNPreprocessor
    from src.models.neural import _build_regressor
    m.pre = NNPreprocessor().fit(policies)
    m.net = _build_regressor(P, m.pre)
    m.net.init_output(0.0, zero_weights=True)
    assert np.allclose(m.predict(policies), glm.predict(policies), rtol=1e-5)


def test_exposure_is_an_offset(policies):
    m = NeuralFrequency(P).fit(policies, policies)
    a = m.predict(policies.assign(Exposure=0.5))
    b = m.predict(policies.assign(Exposure=1.0))
    assert np.allclose(b, 2 * a, rtol=1e-5)


def test_save_load_roundtrip(policies, tmp_path):
    glm = PoissonGLM().fit(policies)
    m = NeuralFrequency(P, glm=glm).fit(policies, policies)
    m.save(tmp_path)
    assert np.allclose(NeuralFrequency.load(tmp_path).predict(policies), m.predict(policies))


def test_autoencoder_scores_and_vae_latent(policies):
    ae = TabularAE(P).fit(policies, policies)
    assert ae.score(policies).shape == (len(policies),) and ae.threshold > 0
    vae = TabularAE({**P, "beta": 1.0}, variational=True).fit(policies, policies)
    assert vae.latent(policies).shape == (len(policies), 2)
