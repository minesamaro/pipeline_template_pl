from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torch.nn.functional as F
import torch

from src.modules.data.metadataframe.metadataframe import MetadataFrame

class MultitaskLossFunction(torch.nn.Module):
    def __init__(self, config, experiment_execution_paths):
        super(MultitaskLossFunction, self).__init__()
        self.config = config

        self.weights_surv = self._get_label_weights(experiment_execution_paths, label='label')
        self.weights_stage = self._get_label_weights(experiment_execution_paths, label='stage')
        print(f"Label weights: {self.weights_surv}, Stage weights: {self.weights_stage}")


    def forward(self, surv_output, surv_target, stage_logits, stage_targets, alpha=1, beta=0.5):

        # Survival Loss 
        surv_loss = binary_cross_entropy_with_logits(
            input=surv_output,
            target=surv_target,
            
            weight=self.weights.to(surv_output.device)
        )

        # Stage loss (cross entropy)
        stage_loss = cross_entropy(
            input=stage_logits,
            target=stage_targets,
            weight=self.weights.to(stage_logits.device)
        )

        loss = alpha * surv_loss + beta * stage_loss

        return loss, surv_loss, stage_loss

    def _get_label_weights(self, experiment_execution_paths, label='label'):
        if self.config.apply_weights:
            metadataframe = MetadataFrame(
                config=self.config.metadataframe,
                experiment_execution_paths=experiment_execution_paths
            )
            lung_nodule_metadataframe = \
                metadataframe.get_lung_metadataframe()

            label_counts = \
                lung_nodule_metadataframe[label].value_counts().sort_index()
            pos_weight = label_counts[0] / label_counts[1]
            return torch.tensor([pos_weight]).to(self.config.device)
        
        else:
            if label == 'label':
                label_weights = torch.tensor([1.0, 1.0]).to(self.config.device)
            elif label == 'stage':
                label_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.config.device)

        return label_weights