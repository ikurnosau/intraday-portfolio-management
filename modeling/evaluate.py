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
                                        trade_percent: float = 0.05) -> None:
    """Evaluate a trained *multi-asset* PyTorch regressor.

    The model must output a tensor of shape ``(batch, asset)`` (or with an extra
    singleton dimension at the end which will be squeezed).  *y_train* /
    *y_test* as well as *test_return* must have the same layout.

    The metrics mirror :func:`evaluate_torch_regressor` but flatten the asset
    dimension when computing RMSE and trading PnL.
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

    # Flatten asset axis so metrics treat every asset-time pair independently
    train_preds = train_preds_arr.reshape(-1)
    test_preds = test_preds_arr.reshape(-1)

    def _to_np(arr):
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return np.asarray(arr)

    y_train_np = _to_np(y_train).reshape(-1)
    y_test_np = _to_np(y_test).reshape(-1)
    test_return_np = _to_np(test_return).reshape(-1)

    # Thresholds on *flattened* predictions -----------------------------------
    long_thresh = pd.Series(test_preds).quantile(1 - trade_percent)
    short_thresh = pd.Series(test_preds).quantile(trade_percent)

    actions = pd.Series(test_preds).apply(
        lambda p: 1 if p > long_thresh else (-1 if p < short_thresh else 0)
    ).to_numpy()

    train_rmse = root_mean_squared_error(y_train_np, train_preds)
    test_rmse = root_mean_squared_error(y_test_np, test_preds)
    baseline_rmse = root_mean_squared_error(y_test_np, np.full_like(y_test_np, y_test_np.mean()))

    expected_return = np.mean(test_return_np * actions) / (2 * trade_percent)

    print(
        f"Train rmse: {train_rmse}, Test rmse: {test_rmse}, Baseline rmse: {baseline_rmse}\n"
        f"Expected return: {expected_return}, Baseline return: {abs(test_return_np.mean())}, "
        f"Max possible return {abs(test_return_np).mean()}"
    )
