import numpy as np
import torch

import modeling.allocator_evaluation as allocator_evaluation


def test_confidence_grid_uses_best_universe_confidences(monkeypatch) -> None:
    threshold_calls: list[float] = []

    class FakeAllocator:
        def __init__(
            self,
            *,
            select_from_n_best: int,
            confidence_threshold: float,
            **_,
        ):
            self.select_from_n_best = select_from_n_best
            self.confidence_threshold = confidence_threshold

        def to(self, _device):
            return self

    def fake_test_cum_wealth(allocator, *_args, **_kwargs):
        if allocator.confidence_threshold == 0:
            if allocator.select_from_n_best == 4:
                return 2.0, None, None, np.array([10.0, 20.0])
            return 1.0, None, None, np.array([100.0, 200.0])

        threshold_calls.append(allocator.confidence_threshold)
        return allocator.confidence_threshold, None, None, None

    monkeypatch.setattr(
        allocator_evaluation,
        "SignalPredictorAllocator",
        FakeAllocator,
    )
    monkeypatch.setattr(
        allocator_evaluation,
        "test_cum_wealth",
        fake_test_cum_wealth,
    )

    best_universe, _ = allocator_evaluation.find_best_allocator_params(
        signal_predictor=torch.nn.Identity(),
        loader=None,
        next_returns=np.zeros((1, 4)),
        spreads=np.zeros((1, 4)),
        volatilities=np.zeros((1, 4)),
        trade_asset_count=1,
        fee=0.0,
        spread_multiplier=1.0,
        device=torch.device("cpu"),
        allow_short_positions=True,
        horizon=1,
        n_runs_per_param=2,
    )

    assert best_universe == 4
    np.testing.assert_allclose(threshold_calls, [11.0, 19.0])
