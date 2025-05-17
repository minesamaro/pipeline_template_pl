import monai.networks.nets as monai_nets
import torch


class ResNet503DModel(torch.nn.Module):
    def __init__(self, config):
        super(ResNet503DModel, self).__init__()

        self.config = config
        self.num_classes = config.number_of_classes
        self.model = monai_nets.resnet50(pretrained=False)
        return

    def forward(self, model_input):
        print(f"[DEBUG] Model input shape: {model_input.shape}")
        return self.model(model_input)
