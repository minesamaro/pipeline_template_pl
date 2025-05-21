from torch.nn import CrossEntropyLoss
import torch

from src.modules.data.metadataframe.metadataframe import MetadataFrame

class CELossFunction(torch.nn.Module):
    def __init__(self, config, experiment_execution_paths):
        super(CELossFunction, self).__init__()
        self.config = config

        self.weights = self._get_label_weights(experiment_execution_paths)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        loss = CrossEntropyLoss(
            weight=self.weights,
            reduction='mean'
        )
        loss = loss(
            input=logits,
            target=targets.squeeze().long()
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
            label_weights = torch.tensor(
                (label_counts.min() / label_counts).tolist()
            ).to(self.config.device)
        else:
            label_weights = torch.tensor([1.0, 1.0]).to(self.config.device)
        return label_weights