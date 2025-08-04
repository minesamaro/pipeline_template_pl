from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch
import wandb
from sklearn.metrics import balanced_accuracy_score

from src.modules.model.vgg163d.vgg16_model \
    import VGG16_3DModel
from src.modules.loss_functions.resnet50_loss_functions \
    import ResNet50LossFunction


class PyTorchLightningVGG163dModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths, test_dataloader= None):
        super().__init__()
        self.config = config

        self.criterion = ResNet50LossFunction(
            config=self.config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        self.labels = None
        self.model = VGG16_3DModel(config=self.config.model)
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
        test_binary_thr025 = (test_preds > 0.25).int()
        test_binary_thr075 = (test_preds > 0.75).int()
        test_binary_thr05 = (test_preds > 0.5).int()

        # Print imbalance information
        #self.print_inbalance(test_binary_preds, test_labels, stage_name="Test Set")


        # Compute test metrics (customize as needed)
        metrics_for_logging = {
            'test_accuracy0.5': accuracy(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_auroc': auroc(
                preds=test_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_precision0.5': precision(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_recall0.5': recall(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_balanced_accuracy0.5': balanced_accuracy_score(
                y_true=test_labels.cpu().numpy(),
                y_pred=test_binary_preds.cpu().numpy()
            ),
            'test_accuracy0.25': accuracy(
                preds=test_binary_thr025,
                target=test_labels.int(),
                task='binary').item(),
            'test_accuracy0.75': accuracy(
                preds=test_binary_thr075,
                target=test_labels.int(),
                task='binary').item(),
            'test_precision0.25': precision(
                preds=test_binary_thr025,
                target=test_labels.int(),
                task='binary').item(),
            'test_precision0.75': precision(
                preds=test_binary_thr075,
                target=test_labels.int(),
                task='binary').item(),
            'test_recall0.25': recall(
                preds=test_binary_thr025,
                target=test_labels.int(),
                task='binary').item(),
            'test_recall0.75': recall(
                preds=test_binary_thr075,
                target=test_labels.int(),
                task='binary').item(),
            'test_balanced_accuracy0.25': balanced_accuracy_score(
                y_true=test_labels.cpu().numpy(),
                y_pred=test_binary_thr025.cpu().numpy()
            ),
            'test_balanced_accuracy0.75': balanced_accuracy_score(
                y_true=test_labels.cpu().numpy(),
                y_pred=test_binary_thr075.cpu().numpy()
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
        test_binary_preds = (predicted_labels > 0.5).int()
        test_binary_thr025 = (predicted_labels > 0.25).int()
        test_binary_thr075 = (predicted_labels > 0.75).int()

        # Print imbalance information
        #self.print_inbalance(predicted_activated_labels, labels, stage_name="Validation Set")
        metrics_for_logging = {
            'val_accuracy0.5': accuracy(
                preds=test_binary_preds,
                target=labels.int(),
                task='binary').item(),
            'val_auroc': auroc(
                preds=predicted_labels,
                target=labels.int(),
                task='binary').item(),
            'val_precision0.5': precision(
                preds=test_binary_preds,
                target=labels.int(),
                task='binary').item(),
            'val_recall0.5': recall(
                preds=test_binary_preds,
                target=labels.int(),
                task='binary').item(),
            'val_balanced_accuracy0.5': balanced_accuracy_score(
                y_true=labels.cpu().numpy(),
                y_pred=test_binary_preds.cpu().numpy()
            ),
            'val_accuracy0.25': accuracy(
                preds=test_binary_thr025,
                target=labels.int(),
                task='binary').item(),
            'val_accuracy0.75': accuracy(
                preds=test_binary_thr075,
                target=labels.int(),
                task='binary').item(),
            'val_precision0.25': precision(
                preds=test_binary_thr025,
                target=labels.int(),
                task='binary').item(),
            'val_precision0.75': precision(
                preds=test_binary_thr075,
                target=labels.int(),
                task='binary').item(),
            'val_recall0.25': recall(
                preds=test_binary_thr025,
                target=labels.int(),
                task='binary').item(),
            'val_recall0.75': recall(
                preds=test_binary_thr075,
                target=labels.int(),
                task='binary').item(),
            'val_balanced_accuracy0.25': balanced_accuracy_score(
                y_true=labels.cpu().numpy(),
                y_pred=test_binary_thr025.cpu().numpy()
            ),
            'val_balanced_accuracy0.75': balanced_accuracy_score(
                y_true=labels.cpu().numpy(),
                y_pred=test_binary_thr075.cpu().numpy()
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
        test_labels = torch.cat(self.labels, dim=0).to(self.device)
        test_preds = torch.cat(self.predicted_labels, dim=0).to(self.device)

        # Print imbalance information
        #self.print_inbalance(predicted_activated_labels, labels, stage_name="Test Set")

        test_binary_preds = (test_preds > 0.5).int()
        test_binary_thr025 = (test_preds > 0.25).int()
        test_binary_thr075 = (test_preds > 0.75).int()

        # Print imbalance information
        #self.print_inbalance(test_binary_preds, test_labels, stage_name="Test Set")


        # Compute test metrics (customize as needed)
        metrics_for_logging = {
            'test_accuracy0.5': accuracy(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_auroc': auroc(
                preds=test_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_precision0.5': precision(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_recall0.5': recall(
                preds=test_binary_preds,
                target=test_labels.int(),
                task='binary').item(),
            'test_balanced_accuracy0.5': balanced_accuracy_score(
                y_true=test_labels.cpu().numpy(),
                y_pred=test_binary_preds.cpu().numpy()
            ),
            'test_accuracy0.25': accuracy(
                preds=test_binary_thr025,
                target=test_labels.int(),
                task='binary').item(),
            'test_accuracy0.75': accuracy(
                preds=test_binary_thr075,
                target=test_labels.int(),
                task='binary').item(),
            'test_precision0.25': precision(
                preds=test_binary_thr025,
                target=test_labels.int(),
                task='binary').item(),
            'test_precision0.75': precision(
                preds=test_binary_thr075,
                target=test_labels.int(),
                task='binary').item(),
            'test_recall0.25': recall(
                preds=test_binary_thr025,
                target=test_labels.int(),
                task='binary').item(),
            'test_recall0.75': recall(
                preds=test_binary_thr075,
                target=test_labels.int(),
                task='binary').item(),
            'test_balanced_accuracy0.25': balanced_accuracy_score(
                y_true=test_labels.cpu().numpy(),
                y_pred=test_binary_thr025.cpu().numpy()
            ),
            'test_balanced_accuracy0.75': balanced_accuracy_score(
                y_true=test_labels.cpu().numpy(),
                y_pred=test_binary_thr075.cpu().numpy()
            ),
        }

        wandb.log(metrics_for_logging, step=self.current_epoch)