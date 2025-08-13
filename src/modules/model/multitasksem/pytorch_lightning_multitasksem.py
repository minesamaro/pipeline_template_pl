from torchmetrics.functional import accuracy, auroc, precision, recall
from sklearn.metrics import balanced_accuracy_score
import pytorch_lightning
import torch
import wandb

from src.modules.model.multitasksem.multitasksem import MultiTaskSurvivalStageNet2D
from src.modules.loss_functions.multitask_loss_functions import MultitaskLossFunction


class PyTorchLightningMultitaskSEMModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths, test_dataloader= None):
        super().__init__()
        self.config = config

        self.criterion = MultitaskLossFunction(
            config=self.config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        self.labels = None
        self.model = MultiTaskSurvivalStageNet2D(
            in_channels=1, base_channels=32, stage_classes=7, use_sem=True
        )
        self.predicted_labels = None
        self.predicted_stage_labels = None
        self.weighted_losses = None
        self.test_ref = test_dataloader
        self.stage_num_classes = 7

        self.to(torch.device(self.config.device))
        self.test_dataloader_ref = test_dataloader

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimiser.type)(
            self.parameters(), **self.config.optimiser.kwargs
        )
        return optimizer
    
    def print_inbalance(self, predicted_activated_labels, labels, stage_name=""):
        # Check how many predictions are 0 and 1
        num_pred_0 = (predicted_activated_labels == 0).sum().item()
        num_pred_1 = (predicted_activated_labels == 1).sum().item()

        # Check how many actual labels are 0 and 1
        num_true_0 = (labels == 0).sum().item()
        num_true_1 = (labels == 1).sum().item()

        print(f"Stage: {stage_name}")

        print(f"Predicted class distribution: 0s = {num_pred_0}, 1s = {num_pred_1}")
        print(f"Actual label distribution:    0s = {num_true_0}, 1s = {num_true_1}")

        if num_pred_1 == 0 and num_pred_0 > 0:
            print("⚠️  Model is predicting only class 0 (majority class). It is ignoring the minority class!")
        elif num_pred_0 == 0 and num_pred_1 > 0:
            print("⚠️  Model is predicting only class 1 (minority class). It is ignoring the majority class!")
        else:
            print("✅ Model is predicting both classes.")

        return

    
    def evaluate_on_test_set(self):
        self.model.eval()
        test_labels = []
        test_preds = []
        test_stage_labels = []
        test_stage_preds = []

        with torch.no_grad():
            for batch in self.test_dataloader_ref:
                data, labels, stage_labels = batch
                surv_logits, stage_logits = self.model(data['image'].to(self.device))
                probs = torch.sigmoid(surv_logits)
                stage_probs = torch.softmax(stage_logits, dim=1)

                test_labels.append(labels)
                test_preds.append(probs)
                test_stage_labels.append(stage_labels)
                test_stage_preds.append(stage_probs)

        test_labels = torch.cat(test_labels, dim=0).to(self.device)
        test_preds = torch.cat(test_preds, dim=0).to(self.device)
        test_stage_labels = torch.cat(test_stage_labels, dim=0).to(self.device)
        test_stage_preds = torch.cat(test_stage_preds, dim=0).to(self.device)

        # Considering the model is binary classification with 1 output
        test_binary_preds = (test_preds > 0.5).int()
        test_binary_thr025 = (test_preds > 0.25).int()
        test_binary_thr075 = (test_preds > 0.75).int()
        test_binary_thr05 = (test_preds > 0.5).int()

        # Print imbalance information
        #self.print_inbalance(test_binary_preds, test_labels, stage_name="Test Set")
        # Get the number of the predicted class from the probabilities
        test_stage_pred_classes = test_stage_preds.argmax(dim=1)

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
            # Now for stage
            'stage_test_accuracy0.5': accuracy(
                preds=test_stage_preds,
                target=test_stage_labels.int(),
                task='multiclass',
                num_classes = self.stage_num_classes).item(),
            'stage_test_auroc': auroc(
                preds=test_stage_preds,
                target=test_stage_labels.int(),
                task='multiclass',
                num_classes = self.stage_num_classes).item(),
            'stage_test_precision0.5': precision(
                preds=test_stage_preds,
                target=test_stage_labels.int(),
                task='multiclass',
                num_classes = self.stage_num_classes).item(),
            'stage_test_recall0.5': recall(
                preds=test_stage_preds,
                target=test_stage_labels.int(),
                task='multiclass',
                num_classes = self.stage_num_classes).item(),
            'stage_test_balanced_accuracy0.5': balanced_accuracy_score(
                y_true=test_stage_labels.cpu().numpy(),
                y_pred=test_stage_pred_classes.cpu().numpy()
            ),
        }
        # Log to wandb
        wandb.log(metrics_for_logging, step=self.current_epoch)

    def on_train_epoch_start(self):
        self.labels = []
        self.stage_labels = []
        self.predicted_labels = []
        self.stage_predicted_labels = []
        self.weighted_losses = []
        self.surv_loss = []
        self.stage_loss = []

    def training_step(self, batch):
        data, labels, stage_labels = batch

        surv_logits, stage_logits = self.model(data['image'].to(self.device))
        activated_labels = torch.sigmoid(surv_logits)
        activated_stage_labels = torch.softmax(stage_logits, dim=1)
        loss, surv_loss, stage_loss = self.criterion(
            surv_output=surv_logits,
            surv_target=labels,
            stage_logits=stage_logits,
            stage_targets=stage_labels
        )

        batch_size = data['image'].shape[0]
        self.labels.append(labels)
        self.stage_labels.append(stage_labels)
        self.predicted_labels.append(activated_labels)
        self.stage_predicted_labels.append(activated_stage_labels)
        self.weighted_losses.append(loss * batch_size)
        self.surv_loss.append(surv_loss * batch_size)
        self.stage_loss.append(stage_loss * batch_size)

        return loss

    def on_train_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        predicted_labels = torch.cat(self.predicted_labels, dim=0)
        stage_labels = torch.cat(self.stage_labels, dim=0)
        predicted_stage_labels = torch.cat(self.stage_predicted_labels, dim=0)

        metrics_for_logging = {
            'train_loss': (sum(self.weighted_losses) / labels.shape[0]).item(),
            'train_auroc': auroc(
                preds=predicted_labels,
                target=labels.int(),
                task='binary').item(),
            'stage_train_auroc': auroc(
                preds=predicted_stage_labels,
                target=stage_labels.int(),
                task='multiclass',
                num_classes=self.stage_num_classes).item(),
            'train_surv_loss': (sum(self.surv_loss) / labels.shape[0]).item(),
            'train_stage_loss': (sum(self.stage_loss) / labels.shape[0]).item(),
        }
        wandb.log(metrics_for_logging, step=self.current_epoch)
        self.log_dict(
            metrics_for_logging,
            batch_size=labels.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=False
        )

        if self.test_dataloader_ref is not None:
            self.evaluate_on_test_set()

        
    def on_validation_epoch_start(self):
        self.val_labels = []
        self.val_predicted_labels = []
        self.val_stage_labels = []
        self.val_stage_predicted_labels = []
        self.val_weighted_losses = []
        self.val_surv_loss = []
        self.val_stage_loss = []

    def validation_step(self, batch, batch_idx):
        data, labels, stage_labels = batch

        surv_logits, stage_logits = self.model(data['image'].to(self.device))
        activated_labels = torch.sigmoid(surv_logits)
        activated_stage_labels = torch.softmax(stage_logits, dim=1)
        loss, surv_loss, stage_loss = self.criterion(
            surv_output=surv_logits,
            surv_target=labels,
            stage_logits=stage_logits,
            stage_targets=stage_labels
        )

        self.val_labels.append(labels)
        self.val_predicted_labels.append(activated_labels)
        self.val_stage_labels.append(activated_stage_labels)
        self.val_stage_predicted_labels.append(activated_stage_labels)
        self.val_weighted_losses.append(loss * data['image'].shape[0])
        self.val_surv_loss.append(surv_loss * data['image'].shape[0])
        self.val_stage_loss.append(stage_loss * data['image'].shape[0])

    def on_validation_epoch_end(self):
        labels = torch.cat(self.val_labels, dim=0).to(self.device)
        predicted_labels = torch.cat(self.val_predicted_labels, dim=0).to(self.device)
        stage_labels = torch.cat(self.val_stage_labels, dim=0).to(self.device)
        predicted_stage_labels = torch.cat(self.val_stage_predicted_labels, dim=0).to(self.device)

        predicted_activated_labels = (predicted_labels > 0.5).int()
        val_stage_pred_classes = predicted_stage_labels.argmax(dim=1)
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
            # now for stage
            'val_stage_accuracy0.5': accuracy(
                preds=val_stage_pred_classes,
                target=stage_labels.int(),
                task='multiclass',
                num_classes=self.num_classes).item(),
            'val_stage_auroc': auroc(
                preds=predicted_stage_labels,
                target=stage_labels.int(),
                task='multiclass',
                num_classes=self.num_classes).item(),
            'val_stage_precision0.5': precision(
                preds=val_stage_pred_classes,
                target=stage_labels.int(),
                task='multiclass',
                num_classes=self.num_classes).item(),
            'val_stage_recall0.5': recall(
                preds=val_stage_pred_classes,
                target=stage_labels.int(),
                task='multiclass',
                num_classes=self.num_classes).item(),
            'val_stage_balanced_accuracy0.5': balanced_accuracy_score(
                y_true=stage_labels.cpu().numpy(),
                y_pred=val_stage_pred_classes.cpu().numpy()
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

        data, labels, stage_labels = batch

        surv_logits, stage_logits = self.model(data['image'].to(self.device))
        activated_labels = torch.sigmoid(surv_logits)
        activated_stage_labels = torch.softmax(stage_logits, dim=1)
        loss, surv_loss, stage_loss = self.criterion(
            surv_output=surv_logits,
            surv_target=labels,
            stage_logits=stage_logits,
            stage_targets=stage_labels
        )

        self.labels.append(labels)
        self.predicted_labels.append(activated_labels)

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
        }

        wandb.log(metrics_for_logging, step=self.current_epoch)