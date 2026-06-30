"""Model output contracts from SPEC.md §5."""

import pytest

torch = pytest.importorskip("torch")  # skip cleanly if torch isn't installed

from src.model import MultiHorizonTCN


def _model(num_features=8, horizons=(1, 5, 20), num_classes=3):
    return MultiHorizonTCN(
        num_features=num_features,
        horizons=horizons,
        num_classes=num_classes,
        channels=(8, 8),
        kernel_size=3,
        dropout=0.0,
    )


def test_forward_returns_one_logit_tensor_per_horizon():
    model = _model()
    x = torch.randn(4, 8, 16)                        # (B, num_features, T)
    out = model(x)
    assert isinstance(out, list) and len(out) == 3
    for logits in out:
        assert logits.shape == (4, 3)


def test_predict_proba_sums_to_one():
    model = _model()
    x = torch.randn(4, 8, 16)
    probs = model.predict_proba(x)
    assert len(probs) == 3
    for p in probs:
        assert p.shape == (4, 3)
        assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert (p >= 0).all()


def test_handles_variable_horizons_and_classes():
    model = _model(num_features=5, horizons=(1, 10), num_classes=2)
    out = model(torch.randn(2, 5, 20))
    assert len(out) == 2
    assert all(o.shape == (2, 2) for o in out)


def test_backward_produces_finite_grads():
    model = _model()
    x = torch.randn(4, 8, 16)
    loss = sum(o.pow(2).mean() for o in model(x))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)
