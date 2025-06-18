import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from typing import Callable
import logging


class Trainer: 
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: torch.nn.Module,
                 optimizer: Optimizer,
                 scheduler: LRScheduler,
                 num_epochs: int=25,
                 device: torch.device=None,
                 metrics: dict[str, Callable]=None,
                 save_path: str=None): 
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.metrics = metrics
        self.save_path = save_path

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)

    def train(self): 
        history = {"train_loss": [], "val_loss": []}
        best_loss = float('inf')

        if self.metrics:
            for name in self.metrics:
                history[f"train_{name}"] = []
                history[f"val_{name}"] = []

        for epoch in range(1, self.num_epochs + 1):
            logging.info(f"Epoch {epoch}/{self.num_epochs}")

            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.evaluate()

            if self.scheduler is not None:
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
            logging.info("")

            # Save model
            if self.save_path and val_loss < best_loss:
                best_loss = val_loss

                torch.save(self.model.state_dict(), self.save_path)
                logging.info(f"Best model saved to {self.save_path} with loss value: {best_loss:.4f}\n")

        return self.model, history

    def train_epoch(self):
        self.model.train()

        num_batches = len(self.train_loader)
        total_loss = 0
        total_metrics = {name: 0.0 for name in (self.metrics or {})}

        for inputs, targets in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.metrics:
                for name, fn in self.metrics.items():
                    total_metrics[name] += fn(outputs, targets)

        print(self.model.training)        # should be True during train(), False during eval()
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                print("BN momentum:", m.momentum)

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
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()

                if self.metrics:
                    for name, fn in self.metrics.items():
                        total_metrics[name] += fn(outputs, targets)

            print(self.model.training)        # should be True during train(), False during eval()
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
