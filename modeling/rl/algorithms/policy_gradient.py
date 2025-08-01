from __future__ import annotations
from typing import List, Dict
import logging

import torch
from tqdm import tqdm

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
        realized_returns: List[float] = []

        with torch.enable_grad() if training else torch.no_grad():
            for (
                signal_features_trajectory_batch,
                next_returns_trajectory_batch,
                spreads_trajectory_batch,
            ) in tqdm(
                data_loader,
                desc=("Train" if training else "Val") + f" Epoch {epoch + 1}/{epochs}",
                leave=False,
            ):
                signal_features_trajectory_batch = signal_features_trajectory_batch.to(self.device, non_blocking=True)
                next_returns_trajectory_batch = next_returns_trajectory_batch.to(self.device, non_blocking=True)
                spreads_trajectory_batch = spreads_trajectory_batch.to(self.device, non_blocking=True)

                trajectory = self.agent.generate_trajectory(
                    signal_features_trajectory_batch=signal_features_trajectory_batch,
                    next_returns_trajectory_batch=next_returns_trajectory_batch,
                    spreads_trajectory_batch=spreads_trajectory_batch,
                )

                if not trajectory:
                    continue

                rewards = [step[2] for step in trajectory]  # list[(batch_size,)]

                realized_batch = (
                    torch.cat(rewards, dim=0).detach().cpu().tolist()
                )  # flatten
                realized_returns.extend(realized_batch)

                loss = self.loss_fn(rewards)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

        return epoch_loss, realized_returns

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
            epoch_loss, realized_returns = self.train_epoch(epoch)

            # --- Validation phase ---
            epoch_loss, realized_returns = self.eval_epoch(epoch)

            # Step the LR scheduler *once per epoch* (if any)
            if self.scheduler is not None:
                self.scheduler.step()

        return self.train_history, self.val_history

    def evaluate(self, actor: BaseActor | None = None) -> tuple[float, List[float]]:
        if actor is not None:
            self.agent.actor = actor.to(self.device)
        epoch_loss, realized_returns = self.eval_epoch(-1)
        return epoch_loss, realized_returns

    def eval_epoch(self, epoch: int) -> tuple[float, List[float]]:
        if self.val_loader is None:
            logging.warning("[PolicyGradient] eval_epoch called but no val_loader was provided.")
            return {}

        # Evaluation mode
        self.agent.actor.eval()

        epoch_loss, realized_returns = self._run_epoch(
            data_loader=self.val_loader,
            epoch=epoch,
            epochs=self.num_epochs,
            training=False,
        )

        print(f"[PolicyGradient] [VAL] Epoch {epoch + 1}/{self.num_epochs} — Loss: {epoch_loss:.4f}")

        if realized_returns:
            epoch_metrics = self.metrics_calculator(realized_returns)
            self.val_history.append(epoch_metrics)
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
            logging.info(f"[PolicyGradient] [VAL] Epoch {epoch + 1}/{self.num_epochs} — {metrics_str}")

        return epoch_loss, realized_returns

    def train_epoch(self, epoch: int) -> tuple[float, List[float]]:
        # Training mode
        self.agent.actor.train()

        epoch_loss, realized_returns = self._run_epoch(
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

        return epoch_loss, realized_returns