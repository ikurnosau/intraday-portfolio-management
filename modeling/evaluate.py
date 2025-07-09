import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, root_mean_squared_error
import lightgbm as lgb
from typing import Union


def _to_tensor(array: Union[np.ndarray, torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Utility: ensure *array* is a torch.Tensor on *device* with *dtype*."""
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    return torch.as_tensor(array, device=device, dtype=dtype)


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def evaluate_lgb_regressor(X_train, y_train, X_test, y_test, test_return, trade_percent=0.05):
    model = lgb.LGBMRegressor(
    n_estimators=1000, 
    learning_rate=0.01,
    max_depth=15,
    num_leaves=31, 
    ).fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    long_treshold = pd.Series(test_preds).quantile(1 - trade_percent)
    short_treshold = pd.Series(test_preds).quantile(trade_percent)
    actions_by_test_preds = pd.Series(test_preds).apply(lambda pred: 1 if pred > long_treshold else -1 if pred < short_treshold else 0).to_numpy()

    train_rmse = root_mean_squared_error(y_train, train_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    baseline_rmse = root_mean_squared_error(y_test, np.zeros_like(y_test) + y_test.mean())
    expected_return = np.mean(test_return * actions_by_test_preds) / (2 * trade_percent)

    print(f'Train rmse: {train_rmse}, Test rmse: {test_rmse}, Baseline rmse: {baseline_rmse}\nExpected return: {expected_return}, Baseline return: {abs(test_return.mean())}, Max possible return {abs(test_return).mean()}')


def evaluate_torch_regressor(model: torch.nn.Module,
                             X_train: Union[np.ndarray, torch.Tensor],
                             y_train: Union[np.ndarray, torch.Tensor],
                             X_test: Union[np.ndarray, torch.Tensor],
                             y_test: Union[np.ndarray, torch.Tensor],
                             test_return: Union[np.ndarray, torch.Tensor],
                             trade_percent: float = 0.05) -> None:
    # ------------------------------------------------------------------
    # Prepare data & model
    # ------------------------------------------------------------------
    model.eval()
    device = next(model.parameters()).device

    X_train_tensor = _to_tensor(X_train, device, torch.float32)
    X_test_tensor = _to_tensor(X_test, device, torch.float32)

    with torch.no_grad():
        train_preds_tensor = model(X_train_tensor)
        test_preds_tensor = model(X_test_tensor)

    # Ensure predictions are 1-D numpy arrays
    train_preds = train_preds_tensor.squeeze(-1).cpu().numpy()
    test_preds = test_preds_tensor.squeeze(-1).cpu().numpy()

    # Convert ground truth & returns to numpy -------------------------------------------------
    y_train_np = _to_numpy(y_train)
    y_test_np = _to_numpy(y_test)
    test_return_np = _to_numpy(test_return)

    # ------------------------------------------------------------------
    # Trading strategy based on prediction quantiles
    # ------------------------------------------------------------------
    long_thresh = pd.Series(test_preds).quantile(1 - trade_percent)
    short_thresh = pd.Series(test_preds).quantile(trade_percent)

    actions = pd.Series(test_preds).apply(
        lambda p: 1 if p > long_thresh else (-1 if p < short_thresh else 0)
    ).to_numpy()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    train_rmse = root_mean_squared_error(y_train_np, train_preds)
    test_rmse = root_mean_squared_error(y_test_np, test_preds)
    baseline_rmse = root_mean_squared_error(y_test_np, np.full_like(y_test_np, y_test_np.mean()))

    expected_return = np.mean(test_return_np * actions) / (2 * trade_percent)

    print(
        f"Train rmse: {train_rmse}, Test rmse: {test_rmse}, Baseline rmse: {baseline_rmse}\n"
        f"Expected return: {expected_return}, Baseline return: {abs(test_return_np.mean())}, "
        f"Max possible return {abs(test_return_np).mean()}"
    )


def evaluate_torch_regressor_multiasset(model: torch.nn.Module,
                                        X_train: Union[np.ndarray, torch.Tensor],
                                        y_train: Union[np.ndarray, torch.Tensor],
                                        X_test: Union[np.ndarray, torch.Tensor],
                                        y_test: Union[np.ndarray, torch.Tensor],
                                        test_return: Union[np.ndarray, torch.Tensor],
                                        test_spread: Union[np.ndarray, torch.Tensor],
                                        trade_asset_count: int = 5) -> None:
    """Evaluate a trained *multi-asset* PyTorch regressor.

    The model must output a tensor of shape ``(batch, asset)`` (or with an extra
    singleton dimension at the end which will be squeezed).  *y_train* /
    *y_test* as well as *test_return* / *test_spread* must have the same layout.

    For every time step (row in the batch dimension) the ``trade_asset_count``
    assets whose predictions deviate the most (in absolute terms) from the
    neutral 0.5 value are selected.  Metrics (RMSE) and the trading PnL are
    then computed **only for these chosen assets**.  All selected assets are
    traded: a *long* position if the prediction is above 0.5, otherwise a
    *short* position.  No additional thresholding is applied.
    """

    # ------------------------------------------------------------------
    # Inference helper to avoid loading the whole set onto the GPU
    # ------------------------------------------------------------------
    def _predict_in_batches(X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """Run *model* on *X* in mini-batches and return predictions as numpy."""

        preds: list[np.ndarray] = []
        n = len(X)
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = X[i : i + batch_size]
                batch_t = _to_tensor(batch, device, torch.float32)
                out = model(batch_t)
                # squeeze regression singleton dim if present
                if out.ndim == 3 and out.shape[-1] == 1:
                    out = out.squeeze(-1)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0)

    # ------------------------------------------------------------------
    # Run batched predictions
    # ------------------------------------------------------------------
    train_preds_arr = _predict_in_batches(X_train)
    test_preds_arr = _predict_in_batches(X_test)

    # ------------------------------------------------------------------
    # Select the *trade_asset_count* most confident assets per time step
    # ------------------------------------------------------------------
    def _topk_indices(a: np.ndarray, k: int) -> np.ndarray:
        """Return indices of the *k* largest values per row (last axis)."""
        # argsort ascending -> take last k columns
        return np.argsort(a, axis=1)[:, -k:]

    abs_dev_train = np.abs(train_preds_arr - 0.5)
    abs_dev_test = np.abs(test_preds_arr - 0.5)

    train_topk_idx = _topk_indices(abs_dev_train, trade_asset_count)  # (batch, k)
    test_topk_idx = _topk_indices(abs_dev_test, trade_asset_count)

    # Gather predictions for selected assets
    rows = np.arange(train_preds_arr.shape[0])[:, None]
    train_preds_sel = train_preds_arr[rows, train_topk_idx]  # (batch, k)
    test_preds_sel = test_preds_arr[np.arange(test_preds_arr.shape[0])[:, None], test_topk_idx]

    # Flatten so each (time, asset) pair becomes one sample for metrics
    train_preds = train_preds_sel.reshape(-1)
    test_preds = test_preds_sel.reshape(-1)

    def _to_np(arr):
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return np.asarray(arr)

    y_train_np_full = _to_np(y_train)
    y_test_np_full = _to_np(y_test)
    test_return_np_full = _to_np(test_return)
    test_spread_np_full = _to_np(test_spread)

    y_train_sel = y_train_np_full[rows, train_topk_idx]
    y_test_sel = y_test_np_full[np.arange(y_test_np_full.shape[0])[:, None], test_topk_idx]
    test_return_sel = test_return_np_full[np.arange(test_return_np_full.shape[0])[:, None], test_topk_idx]
    test_spread_sel = test_spread_np_full[np.arange(test_spread_np_full.shape[0])[:, None], test_topk_idx]

    y_train_np = y_train_sel.reshape(-1)
    y_test_np = y_test_sel.reshape(-1)
    test_return_np = test_return_sel.reshape(-1)
    test_spread_np = test_spread_sel.reshape(-1)

    # ------------------------------------------------------------------
    # Trading strategy: trade every selected asset (direction by sign)
    # ------------------------------------------------------------------
    actions = np.where(test_preds > 0.5, 1, -1)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    train_rmse = root_mean_squared_error(y_train_np, train_preds)
    test_rmse = root_mean_squared_error(y_test_np, test_preds)
    baseline_rmse = root_mean_squared_error(y_test_np, np.full_like(y_test_np, y_test_np.mean()))

    # Deduct transaction cost (spread) from realised returns
    expected_return = np.mean(test_return_np * actions - test_spread_np)
    expected_return_without_spread = np.mean(test_return_np * actions)

    print(
        f"Train rmse: {train_rmse}, Test rmse: {test_rmse}, Baseline rmse: {baseline_rmse}\n"
        f"Expected return: {expected_return}, Expected return without spread: {expected_return_without_spread}, Baseline return: {abs(test_return_np.mean())}, "
        f"Max possible return {abs(test_return_np).mean()}"
    )
