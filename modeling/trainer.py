import torch
import math
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm
from typing import Callable, Union
import logging
import copy  # Local import to avoid polluting global namespace unnecessarily


class Trainer: 
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: torch.nn.Module,
                 optimizer: Optimizer,
                 scheduler: Union[LRScheduler, dict],
                 num_epochs: int=25,
                 early_stopping_patience: int=5,
                 device: torch.device=None,
                 metrics: dict[str, Callable]=None,
                 save_path: str=None): 
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.metrics = metrics
        self.save_path = save_path

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)

        # Automatic Mixed Precision (AMP)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        # ---------------------------------------------
        # Optional Torch 2.0 ahead-of-time compilation
        # ---------------------------------------------
        try:
            if hasattr(torch, "compile") and self.device.type == "cuda":
                # `mode="reduce-overhead"` is the safest default for training
                self.model = torch.compile(self.model, mode="reduce-overhead")  # type: ignore[attr-defined]
                logging.info("Model compiled with torch.compile()")
        except Exception as e:  # pragma: no cover – compilation is optional
            logging.warning(f"torch.compile failed – falling back to eager: {e}")

        # ------------------------------------------------------------------
        # Multi-GPU support: automatically wrap in DataParallel when >1 GPU
        # ------------------------------------------------------------------
        if torch.cuda.device_count() > 1 and self.device.type == "cuda":
            logging.info(f"DataParallel enabled – using {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)
            # Ensure optimizer references the (possibly) wrapped parameters
            # Users typically construct the optimizer *before* Trainer, so its
            # parameter groups already reference the original module.  This is
            # still OK because DataParallel just replicates that module on each
            # forward pass – parameters stay shared.  However, if an optimizer
            # *without* parameters was passed, we re-associate it here.
            if len(self.optimizer.param_groups) == 0:  # pragma: no cover
                self.optimizer.add_param_group({"params": self.model.parameters()})

        # Handle scheduler configuration
        if isinstance(scheduler, dict) and scheduler['type'] == 'OneCycleLR':
            # If OneCycleLR config is passed, create it with correct steps_per_epoch
            scheduler_config = scheduler.copy()
            scheduler_config.pop('type')
            self.scheduler = OneCycleLR(
                optimizer=optimizer,
                steps_per_epoch=len(train_loader),
                epochs=num_epochs,
                **scheduler_config
            )
        else:
            self.scheduler = scheduler

    def train(self): 
        history = {"train_loss": [], "val_loss": []}
        best_loss = float('inf')
        best_epoch = 0
        best_model_state = None  # Will store a deepcopy of the best weights

        if self.metrics:
            for name in self.metrics:
                history[f"train_{name}"] = []
                history[f"val_{name}"] = []

        for epoch in range(1, self.num_epochs + 1):
            logging.info(f"Epoch {epoch}/{self.num_epochs}")

            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.evaluate()

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif isinstance(self.scheduler, OneCycleLR):
                    pass
                else:
                    self.scheduler.step()

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.metrics:
                for name in self.metrics:
                    history[f"train_{name}"].append(train_metrics[name])
                    history[f"val_{name}"].append(val_metrics[name])

            # Print metrics
            logging.info(f"Train Loss: {train_loss:.4f}")
            for name in train_metrics:
                logging.info(f"Train {name.capitalize()}: {train_metrics[name]:.4f}")
            logging.info(f"Val   Loss: {val_loss:.4f}")
            for name in val_metrics:
                logging.info(f"Val   {name.capitalize()}: {val_metrics[name]:.4f}")

            # Save model
            if val_loss < best_loss:
                logging.info("New best model found! Updating best state dict.")
                best_loss = val_loss
                best_epoch = epoch

                # If DataParallel was used, the underlying model weights live
                # in the `.module` attribute.  Save a clean state_dict so the
                # checkpoint can be loaded without DataParallel as well.
                state_dict = (
                    self.model.module.state_dict()
                    if isinstance(self.model, torch.nn.DataParallel)
                    else self.model.state_dict()
                )

                # Keep a local copy of the best weights so we can return the best model
                # after training finishes, without needing to reload from disk.
                best_model_state = copy.deepcopy(state_dict)

                # Persist to disk if a save_path was provided
                if self.save_path:
                    torch.save(state_dict, self.save_path)
                    logging.info(
                        f"Best model saved to {self.save_path} with loss value: {best_loss:.4f}\n"
                    )

            if epoch - best_epoch >= self.early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
                
            logging.info("")

        # After all epochs complete, ensure that the model holds the best-performing weights
        if best_model_state is not None:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(best_model_state)
            else:
                self.model.load_state_dict(best_model_state)

        return self.model, history

    def train_epoch(self):
        self.model.train()

        num_batches = len(self.train_loader)
        total_loss = 0
        total_metrics = {name: 0.0 for name in (self.metrics or {})}

        for inputs, targets in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()
            # Clip gradients (unscaled) – call before scaler.step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            if math.isnan(loss.item()):
                logging.warning('Loss is Nan!')
                logging.info(f'Outputs:\n{outputs}')
                logging.info(f'Targets:\n{targets}')
                1 / 0

            total_loss += loss.item()

            if self.metrics:
                for name, fn in self.metrics.items():
                    total_metrics[name] += fn(outputs, targets)

        epoch_loss = total_loss / num_batches
        epoch_metrics = {name: total_metrics[name] / num_batches for name in total_metrics}
        return epoch_loss, epoch_metrics

    def evaluate(self):
        self.model.eval()

        num_batches = len(self.val_loader)
        total_loss = 0
        total_metrics = {name: 0.0 for name in (self.metrics or {})}

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Evaluating", leave=False):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()

                if self.metrics:
                    for name, fn in self.metrics.items():
                        total_metrics[name] += fn(outputs, targets)

            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    print("BN momentum:", m.momentum)

        epoch_loss = total_loss / num_batches
        epoch_metrics = {name: total_metrics[name] / num_batches for name in total_metrics}
        return epoch_loss, epoch_metrics


if __name__ == "__main__":
    # Example usage:
    # from your_model import MyModel
    # from torch.utils.data import DataLoader
    #
    #
    # model = MyModel()
    # train_loader = DataLoader(...)  # your training dataset
    # val_loader = DataLoader(...)    # your validation dataset
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # metrics = {"accuracy": accuracy, "precision": precision}
    #
    # trained_model, history = train(
    #     model,
    #     train_loader,
    #     val_loader,
    #     loss_fn,
    #     optimizer,
    #     scheduler=scheduler,
    #     num_epochs=20,
    #     metrics=metrics,
    #     save_path="best_model.pth"
    # )
    pass
