from __future__ import annotations
from typing import List, Dict
import logging

import torch
from tqdm import tqdm
import pandas as pd

from ..agent import RlAgent
from ..metrics import MetricsCalculator
from ..actors.base_actor import BaseActor


class PolicyGradient:
    def __init__(
        self,
        agent: RlAgent,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        metrics_calculator: MetricsCalculator,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        num_epochs: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        max_grad_norm: float | None = 1.0,
        device: torch.device | str = "cuda",
    ):
        self.agent = agent
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.metrics_calculator = metrics_calculator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(device)
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

    def _run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        epochs: int,
        training: bool = True,
    ) -> tuple[float, List[float]]:
        epoch_loss = 0.0
        epoch_realized_returns: List[float] = []
        epoch_actions: List[torch.Tensor] = []

        with torch.enable_grad() if training else torch.no_grad():
            for (
                signal_features_trajectory_batch,
                next_returns_trajectory_batch,
                spreads_trajectory_batch,
                volatility_trajectory_batch,
            ) in tqdm(
                data_loader,
                desc=("Train" if training else "Val") + f" Epoch {epoch + 1}/{epochs}",
                leave=False,
            ):
                signal_features_trajectory_batch = signal_features_trajectory_batch.to(self.device, non_blocking=True)
                next_returns_trajectory_batch = next_returns_trajectory_batch.to(self.device, non_blocking=True)
                spreads_trajectory_batch = spreads_trajectory_batch.to(self.device, non_blocking=True)
                volatility_trajectory_batch = volatility_trajectory_batch.to(self.device, non_blocking=True)

                trajectory = self.agent.generate_trajectory(
                    signal_features_trajectory_batch=signal_features_trajectory_batch,
                    next_returns_trajectory_batch=next_returns_trajectory_batch,
                    spreads_trajectory_batch=spreads_trajectory_batch,
                    volatility_trajectory_batch=volatility_trajectory_batch,
                )

                if not trajectory:
                    continue

                rewards = [step[2] for step in trajectory]  # list[(batch_size,)]
                losses = [step[3] for step in trajectory]
                epoch_realized_returns.extend(
                    torch.stack(rewards)\
                        .t()\
                        .reshape(-1)\
                        .detach().cpu().tolist()
                )

                actions = [step[1] for step in trajectory]  # list[(batch_size, n_assets)]
                epoch_actions.extend(
                    torch.stack(actions)\
                        .transpose(0, 1)\
                        .reshape(-1, actions[0].shape[1])\
                        .detach().cpu().tolist()
                )

                loss = self.loss_fn(rewards, losses)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                epoch_loss += loss.item()

        return epoch_loss, epoch_realized_returns, epoch_actions

    def train(self):
        """Run training & evaluation loop.

        Parameters
        ----------
        epochs : int | None, optional
            Number of epochs to run. If *None* (default) the value provided during class
            construction (``num_epochs``) will be used. This makes it possible to specify
            the total number of epochs only once – in the constructor – while still
            giving callers flexibility to override it on demand.
        """

        for epoch in range(self.num_epochs):
            # --- Training phase ---
            epoch_loss, realized_returns, actions = self.train_epoch(epoch)

            # --- Validation phase ---
            epoch_loss, realized_returns, actions = self.eval_epoch(epoch)

            # Step the LR scheduler *once per epoch* (if any)
            if self.scheduler is not None:
                self.scheduler.step()

        return self.train_history, self.val_history

    def evaluate(self, actor: BaseActor | None = None, eval_loader: torch.utils.data.DataLoader | None = None) -> tuple[float, List[float]]:
        if actor is not None:
            self.agent.actor = actor.to(self.device)
            
        if eval_loader is not None:
            self.val_loader = eval_loader

        epoch_loss, realized_returns, actions = self.eval_epoch(-1)
        return epoch_loss, realized_returns, actions

    def eval_epoch(self, epoch: int) -> tuple[float, List[float]]:
        # Evaluation mode
        self.agent.actor.eval()

        epoch_loss, realized_returns, actions = self._run_epoch(
            data_loader=self.val_loader,
            epoch=epoch,
            epochs=self.num_epochs,
            training=False,
        )

        print(f"[PolicyGradient] [VAL] Epoch {epoch + 1}/{self.num_epochs} — Loss: {epoch_loss:.4f}")

        if realized_returns:
            realized_returns = pd.Series(realized_returns).dropna().tolist()
            epoch_metrics = self.metrics_calculator(realized_returns)
            self.val_history.append(epoch_metrics)
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
            logging.info(f"[PolicyGradient] [VAL] Epoch {epoch + 1}/{self.num_epochs} — {metrics_str}")

        return epoch_loss, realized_returns, actions

    def train_epoch(self, epoch: int) -> tuple[float, List[float]]:
        # Training mode
        self.agent.actor.train()

        epoch_loss, realized_returns, actions = self._run_epoch(
            data_loader=self.train_loader,
            epoch=epoch,
            epochs=self.num_epochs,
            training=True,
        )

        print(f"[PolicyGradient] Epoch {epoch + 1}/{self.num_epochs} — Loss: {epoch_loss:.4f}")

        if realized_returns:
            epoch_metrics = self.metrics_calculator(realized_returns)
            self.train_history.append(epoch_metrics)
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
            logging.info(f"[PolicyGradient] Epoch {epoch + 1}/{self.num_epochs} — {metrics_str}")

        return epoch_loss, realized_returns, actions