from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch
import wandb

from src.modules.model.resnet502d.resnet502d_model \
    import ResNet50Model
from src.modules.loss_functions.resnet50_loss_functions \
    import ResNet50LossFunction


class PyTorchLightningResNet502dModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config

        self.criterion = ResNet50LossFunction(
            config=self.config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        self.labels = None
        self.model = ResNet50Model(config=self.config.model)
        self.predicted_labels = None
        self.weighted_losses = None

        self.to(torch.device(self.config.device))

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimiser.type)(
            self.parameters(), **self.config.optimiser.kwargs
        )
        return optimizer

    def on_train_epoch_start(self):
        self.labels = []
        self.predicted_labels = []
        self.weighted_losses = []

    def training_step(self, batch):
        data, labels = batch[0], batch[1]

        model_output = self.model(data['image'].to(self.device))
        predicted_labels = torch.argmax(model_output, dim=1, keepdim=True)
        loss = self.criterion(
            logits=model_output,
            targets=labels.to(self.device)
        )

        self.labels.append(labels)
        self.predicted_labels.append(predicted_labels)
        self.weighted_losses.append(loss * data['image'].shape[0])


        self.log(
            batch_size=data['image'].shape[0],
            name="train_loss",
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            value=loss
        )
   
        return loss
    
    def on_train_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        predicted_labels = torch.cat(self.predicted_labels, dim=0)

        metrics_for_logging = {
            'train_loss': (sum(self.weighted_losses) / labels.shape[0]).item(),
            'train_auroc': auroc(
                preds=predicted_labels.float(),
                target=labels.int(),
                task="binary"
            ).item()
        }
        wandb.log(metrics_for_logging)

        self.log_dict(
            metrics_for_logging,
            batch_size=labels.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=False
        )

        
    def on_validation_epoch_start(self):
        self.labels = []
        self.predicted_labels = []
        self.weighted_losses = []

    def validation_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        model_output = self.model(data['image'].to(self.device))
        predicted_labels = torch.argmax(model_output, dim=1, keepdim=True)
        loss = self.criterion(
            logits=model_output,
            targets=labels.to(self.device)
        )

        self.labels.append(labels)
        self.predicted_labels.append(predicted_labels)
        self.weighted_losses.append(loss * data['image'].shape[0])

    def on_validation_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        predicted_labels = torch.cat(self.predicted_labels, dim=0)

        metrics_for_logging = {
            'val_loss': (sum(self.weighted_losses) / labels.shape[0]).item(),
            'val_accuracy': accuracy(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item(),
            'val_auroc': auroc(
                preds=predicted_labels.float(),
                target=labels.int(),
                task="binary"
            ).item(),
            'val_precision': precision(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item(),
            'val_recall': recall(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item()
        }
        wandb.log(metrics_for_logging)

        self.log_dict(
            metrics_for_logging,
            batch_size=labels.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=False
        )

    def on_test_epoch_start(self):
        self.labels = []
        self.predicted_labels = []

    def test_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        model_output = self.model(data['image'].to(self.device))
        predicted_labels = torch.argmax(model_output, dim=1, keepdim=True)

        self.labels.append(labels)
        self.predicted_labels.append(predicted_labels)

    def on_test_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        predicted_labels = torch.cat(self.predicted_labels, dim=0)

        metrics_for_logging = {
            'test_accuracy': accuracy(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item(),
            'test_auroc': auroc(
                preds=predicted_labels.float(),
                target=labels.int(),
                task="binary"
            ).item(),
            'test_precision': precision(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item(),
            'test_recall': recall(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item()
        }
        self.log_dict(
            metrics_for_logging,
            batch_size=labels.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=False
        )
        wandb.log(metrics_for_logging)
