from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
import torch

from src.modules.data.metadataframe.metadataframe import MetadataFrame

class RegularizationLossFunction(torch.nn.Module):
    def __init__(self, config, experiment_execution_paths):
        super(RegularizationLossFunction, self).__init__()
        self.config = config

        self.weights = self._get_label_weights(experiment_execution_paths)
        print(f"Label weights: {self.weights}")

    def forward(self, surv_output, surv_target, recon_output, recon_target, alpha = 1.0, beta = 0.5):

        

        surv_loss = binary_cross_entropy_with_logits(
            input=surv_output,
            target=surv_target,
            weight=self.weights.to(surv_output.device)
        )

        recon_loss = F.mse_loss(recon_output, recon_target)

        loss = alpha * surv_loss + beta * recon_loss
        return loss

    def _get_label_weights(self, experiment_execution_paths):
        if self.config.apply_weights:
            metadataframe = MetadataFrame(
                config=self.config.metadataframe,
                experiment_execution_paths=experiment_execution_paths
            )
            lung_nodule_metadataframe = \
                metadataframe.get_lung_metadataframe()

            label_counts = \
                lung_nodule_metadataframe['label'].value_counts().sort_index()
            pos_weight = label_counts[0] / label_counts[1]
            return torch.tensor([pos_weight]).to(self.config.device)
        
        else:
            label_weights = torch.tensor([1.0, 1.0]).to(self.config.device)
        return label_weights