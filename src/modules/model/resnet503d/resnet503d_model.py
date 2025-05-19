import monai.networks.nets as monai_nets
import torch


class ResNet503DModel(torch.nn.Module):
    def __init__(self, config):
        super(ResNet503DModel, self).__init__()

        self.config = config
        self.num_classes = config.number_of_classes
        self.model = monai_nets.resnet50(pretrained=False, spatial_dims=3, num_classes=self.num_classes)
        return

    def forward(self, model_input):
        #model_input = model_input.permute(1, 0, 2, 3, 4)
        model_input = model_input.unsqueeze(1)
        if model_input.shape[1] != 3:
            model_input = model_input.repeat(1, 3, 1, 1, 1)

        print(f"[DEBUG] Model input shape: {model_input.shape}")
        return self.model(model_input)
