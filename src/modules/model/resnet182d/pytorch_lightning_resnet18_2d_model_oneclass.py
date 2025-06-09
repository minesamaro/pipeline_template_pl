from torchmetrics.functional import accuracy, auroc, precision, recall
from sklearn.metrics import balanced_accuracy_score
import pytorch_lightning
import torch
import wandb

from src.modules.model.resnet182d.resnet182d_model_oneclass \
    import ResNet18Model
from src.modules.loss_functions.resnet50_loss_functions \
    import ResNet50LossFunction


class PyTorchLightningResNet182dModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths, test_dataloader= None):
        super().__init__()
        self.config = config

        self.criterion = ResNet50LossFunction(
            config=self.config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        self.labels = None
        self.model = ResNet18Model(config=self.config.model)
        self.predicted_labels = None
        self.weighted_losses = None
        self.test_ref = test_dataloader

        self.to(torch.device(self.config.device))
        self.test_dataloader_ref = test_dataloader

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimiser.type)(
            self.parameters(), **self.config.optimiser.kwargs
        )
        return optimizer
    
    def evaluate_on_test_set(self):
        self.model.eval()
        test_labels = []
        test_preds = []

        with torch.no_grad():
            for batch in self.test_dataloader_ref:
                data, labels = batch[0], batch[1]
                outputs = self.model(data['image'].to(self.device))
                probs = torch.sigmoid(outputs)

                test_labels.append(labels)
                test_preds.append(probs)

        test_labels = torch.cat(test_labels, dim=0).to(self.device)
        test_preds = torch.cat(test_preds, dim=0).to(self.device)
        # Considering the model is binary classification with 1 output
        test_binary_preds = (test_preds > 0.5).int()


        # Compute test metrics (customize as needed)
        metrics_for_logging = {
            'test_accuracy': accuracy(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_auroc': auroc(
                preds=test_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_precision': precision(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_recall': recall(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_balanced_accuracy': balanced_accuracy_score(
                y_true=test_labels.cpu().numpy(),
                y_pred=test_binary_preds.cpu().numpy()
            ),
        }
        # Log to wandb
        wandb.log(metrics_for_logging, step=self.current_epoch)

    def on_train_epoch_start(self):
        self.labels = []
        self.predicted_labels = []
        self.weighted_losses = []

    def training_step(self, batch):
        data, labels = batch[0], batch[1]

        model_output = self.model(data['image'].to(self.device))
        activated_labels = torch.sigmoid(model_output)
        loss = self.criterion(
            logits=model_output,
            targets=labels
        )

        self.labels.append(labels)
        self.predicted_labels.append(activated_labels)
        self.weighted_losses.append(loss * data['image'].shape[0]) ## What is this
   
        return loss
    
    def on_train_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        predicted_labels = torch.cat(self.predicted_labels, dim=0)

        metrics_for_logging = {
            'train_loss': (sum(self.weighted_losses) / labels.shape[0]).item(),
            'train_auroc': auroc(
                preds=predicted_labels,
                target=labels.int(),
                task='binary').item()
        }
        wandb.log(metrics_for_logging, step=self.current_epoch)

        if self.test_dataloader_ref is not None:
            self.evaluate_on_test_set()

        
    def on_validation_epoch_start(self):
        self.val_labels = []
        self.val_predicted_labels = []
        self.val_weighted_losses = []

    def validation_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        model_output = self.model(data['image'].to(self.device))
        activated_labels = torch.sigmoid(model_output)
        loss = self.criterion(
            logits=model_output,
            targets=labels
        )

        self.val_labels.append(labels)
        self.val_predicted_labels.append(activated_labels)
        self.val_weighted_losses.append(loss * data['image'].shape[0])

    def on_validation_epoch_end(self):
        labels = torch.cat(self.val_labels, dim=0).to(self.device)
        predicted_labels = torch.cat(self.val_predicted_labels, dim=0).to(self.device)
        predicted_activated_labels = (predicted_labels > 0.5).int()

        metrics_for_logging = {
            'val_loss': (sum(self.val_weighted_losses) / labels.shape[0]).item(),
            'val_accuracy': accuracy(
                preds=predicted_activated_labels,
                target=labels.int(),
                task='binary').item(),
            'val_auroc': auroc(
                preds=predicted_labels,
                target=labels.int(),
                task='binary').item(),
            'val_precision': precision(
                preds=predicted_activated_labels,
                target=labels.int(),
                task='binary').item(),
            'val_recall': recall(
                preds=predicted_activated_labels,
                target=labels.int(),
                task='binary').item(),
            'val_balanced_accuracy': balanced_accuracy_score(
                y_true=labels.cpu().numpy(),
                y_pred=predicted_activated_labels.cpu().numpy()
            ),
        }
        wandb.log(metrics_for_logging, step=self.current_epoch)
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
        predicted_labels = torch.sigmoid(model_output)

        self.labels.append(labels)
        self.predicted_labels.append(predicted_labels)

    def on_test_epoch_end(self):
        labels = torch.cat(self.labels, dim=0).to(self.device)
        predicted_labels = torch.cat(self.predicted_labels, dim=0).to(self.device)
        predicted_activated_labels = (predicted_labels > 0.5).int()

        metrics_for_logging = {
            'test_accuracy': accuracy(
                preds=predicted_activated_labels,
                target=labels.int(),
                task='binary').item(),
            'test_auroc': auroc(
                preds=predicted_labels,
                target=labels.int(),
                task='binary').item(),
            'test_precision': precision(
                preds=predicted_activated_labels,
                target=labels.int(),
                task='binary').item(),
            'test_recall': recall(
                preds=predicted_activated_labels,
                target=labels.int(),
                task='binary').item(),
            'test_balanced_accuracy': balanced_accuracy_score(
                y_true=labels.cpu().numpy(),
                y_pred=predicted_activated_labels.cpu().numpy()
            ),
        }

        wandb.log(metrics_for_logging, step=self.current_epoch)