from torchvision.models import resnet50
import torch


class ResNet50Model(torch.nn.Module):
    def __init__(self, config):
        super(ResNet50Model, self).__init__()
        self.model = resnet50(weights=None)

        if config.dimension == 2.5:
            self.model.conv1 = torch.nn.Conv2d(
                in_channels=10, #TODO: Change this to make more customizable
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False
            )
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            1
        )

    def forward(self, model_input):
        if model_input.shape[1] == 1:
            # If the input has only one channel, repeat it to make it 3 channels
            # This is necessary because ResNet50 expects 3-channel input
            # (e.g., RGB images)
            model_input = model_input.repeat(1, 3, 1, 1)
        model_output = self.model(model_input)
        return model_output
