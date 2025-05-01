import monai.networks.nets as monai_nets
import torch


class ResNet503DModel(torch.nn.Module):
    def __init__(self, config):
        super(ResNet503DModel, self).__init__()

        self.model = monai_nets.ResNet(
            block_type='BASIC',
            n_input_channels=1,
            n_classes=self.num_classes,
            spatial_dims=3,
            num_blocks=(3, 4, 6, 3)  # ResNet50-like
        )
        return self.model

    def forward(self, model_input):
        return self.model(model_input)
