from torch.nn.functional import binary_cross_entropy_with_logits
import torch

from src.modules.data.metadataframe.metadataframe import MetadataFrame

class ResNet50LossFunction(torch.nn.Module):
    def __init__(self, config, experiment_execution_paths):
        super(ResNet50LossFunction, self).__init__()
        self.config = config

        self.weights = self._get_label_weights(experiment_execution_paths)
        print(f"Label weights: {self.weights}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        #loss = binary_cross_entropy_with_logits(
        #    input=logits,
        #    target=torch.nn.functional.one_hot(
        #        targets.squeeze().long(),
        #        num_classes=2
        #    ).float(),
        #    weight=self.weights
        #)
        
        loss = binary_cross_entropy_with_logits(
            input=logits,
            target=targets,
            weight=self.weights.to(logits.device)
        )
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