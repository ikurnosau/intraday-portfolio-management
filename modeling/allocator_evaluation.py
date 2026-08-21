import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core_inference.allocators.signal_predictor_allocator import (
    SignalPredictorAllocator,
)


def calc_realized_returns(
    allocations: np.ndarray,
    next_returns: np.ndarray,
    spreads: np.ndarray,
    fee: float,
    spread_multiplier: float,
) -> np.ndarray:
    allocations_with_initial = np.vstack([
        np.zeros_like(allocations[:1]),
        allocations,
    ])
    realized_returns = []
    for prev_allocation, allocation, next_return, spread in zip(
        allocations_with_initial[:-1],
        allocations_with_initial[1:],
        next_returns,
        spreads,
    ):
        return_component = allocation * next_return
        cost_component = np.abs(prev_allocation - allocation) * (
            fee + (spread / 2) * spread_multiplier
        )
        realized_returns.append((return_component - cost_component).sum())

    return np.asarray(realized_returns)


def get_allocations(
    portfolio_allocator: torch.nn.Module,
    test_loader: DataLoader,
    spreads: np.ndarray,
    volatilities: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if torch.cuda.is_available() and bool(
        int(os.getenv("ENABLE_TORCH_COMPILE", "0"))
    ):
        try:
            portfolio_allocator = torch.compile(
                portfolio_allocator,
                mode="reduce-overhead",
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.warning(
                "torch.compile unavailable, using eager mode: %s",
                exc,
            )
    portfolio_allocator.eval()

    all_allocations, all_confidences = [], []
    offset = 0
    for inputs, targets in tqdm(test_loader, desc="Testing", leave=False):
        batch_size = inputs.shape[0]
        spread_batch = torch.as_tensor(
            spreads[offset:offset + batch_size],
            device=device,
        )
        volatility_batch = torch.as_tensor(
            volatilities[offset:offset + batch_size],
            device=device,
        )
        offset += batch_size

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available(),
        ):
            allocations, confidences = portfolio_allocator(
                inputs,
                spread_batch,
                volatility_batch,
            )
            all_allocations.append(allocations.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy().flatten())

    return (
        np.vstack(all_allocations),
        np.concatenate(all_confidences).flatten(),
    )


def test_cum_wealth(
    allocator: torch.nn.Module,
    test_loader: DataLoader,
    next_returns: np.ndarray,
    spreads: np.ndarray,
    volatilities: np.ndarray,
    *,
    fee: float,
    spread_multiplier: float,
    device: torch.device,
    horizon: int = 1,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    allocations, confidences = get_allocations(
        allocator,
        test_loader,
        spreads,
        volatilities,
        device,
    )
    allocations = allocations[::horizon]
    confidences = confidences[::horizon]
    realized_returns = calc_realized_returns(
        allocations,
        next_returns[::horizon],
        spreads[::horizon],
        fee,
        spread_multiplier,
    )
    cumulative_wealth = float(np.prod(1.0 + realized_returns) - 1.0)
    return cumulative_wealth, realized_returns, allocations, confidences


def find_best_allocator_params(
    signal_predictor: torch.nn.Module,
    loader: DataLoader,
    next_returns: np.ndarray,
    spreads: np.ndarray,
    volatilities: np.ndarray,
    *,
    trade_asset_count: int,
    fee: float,
    spread_multiplier: float,
    device: torch.device,
    allow_short_positions: bool,
    horizon: int,
    n_runs_per_param: int = 20,
) -> tuple[int, float]:
    best_cum_wealth = -100.0
    best_select_from_n_best = 0
    best_confidences: np.ndarray | None = None
    n_assets = spreads.shape[1]
    step = max(1, n_assets // (n_runs_per_param * 2))
    select_from_n_best_range = range(
        n_assets,
        n_assets // 2,
        -step,
    )
    for select_from_n_best in select_from_n_best_range:
        logging.info(
            "Running with select_from_n_best: %s",
            select_from_n_best,
        )
        allocator = SignalPredictorAllocator(
            signal_predictor=signal_predictor,
            trade_asset_count=trade_asset_count,
            select_from_n_best=select_from_n_best,
            confidence_threshold=0,
            allow_short_positions=allow_short_positions,
        ).to(device)
        cum_wealth, _, _, confidences = test_cum_wealth(
            allocator,
            loader,
            next_returns,
            spreads,
            volatilities,
            fee=fee,
            spread_multiplier=spread_multiplier,
            device=device,
            horizon=horizon,
        )
        logging.info("Obtained cum wealth: %s", cum_wealth)
        if cum_wealth > best_cum_wealth:
            best_cum_wealth = cum_wealth
            best_select_from_n_best = select_from_n_best
            best_confidences = confidences
            logging.info(
                "New best select_from_n_best: %s, cum_wealth: %s",
                best_select_from_n_best,
                best_cum_wealth,
            )

    logging.info(
        "Best select_from_n_best: %s\n",
        best_select_from_n_best,
    )

    best_confidence_threshold = 0.0
    best_cum_wealth = -100.0
    if best_confidences is None:
        raise RuntimeError("No allocator universe candidates were evaluated")
    q10 = np.quantile(best_confidences, 0.10)
    q90 = np.quantile(best_confidences, 0.90)
    for confidence_threshold in np.linspace(q10, q90, n_runs_per_param):
        logging.info(
            "Running with select_from_n_best: %s and confidence threshold: %s",
            best_select_from_n_best,
            confidence_threshold,
        )
        allocator = SignalPredictorAllocator(
            signal_predictor=signal_predictor,
            trade_asset_count=trade_asset_count,
            select_from_n_best=best_select_from_n_best,
            confidence_threshold=confidence_threshold,
            allow_short_positions=allow_short_positions,
        ).to(device)
        cum_wealth, _, _, _ = test_cum_wealth(
            allocator,
            loader,
            next_returns,
            spreads,
            volatilities,
            fee=fee,
            spread_multiplier=spread_multiplier,
            device=device,
            horizon=horizon,
        )
        logging.info("Obtained cum wealth: %s", cum_wealth)
        if cum_wealth > best_cum_wealth:
            best_cum_wealth = cum_wealth
            best_confidence_threshold = confidence_threshold
            logging.info(
                "New best confidence threshold: %s, cum_wealth: %s",
                best_confidence_threshold,
                best_cum_wealth,
            )

    logging.info(
        "Best confidence threshold: %s",
        best_confidence_threshold,
    )
    return best_select_from_n_best, float(best_confidence_threshold)
