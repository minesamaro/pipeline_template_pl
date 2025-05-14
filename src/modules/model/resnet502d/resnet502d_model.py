from torchvision.models import resnet50
import torch


class ResNet50Model(torch.nn.Module):
    def __init__(self, config):
        super(ResNet50Model, self).__init__()
        self.model = resnet50(weights=None)
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        if model_input.shape[1] == 1:
            # If the input has only one channel, repeat it to make it 3 channels
            # This is necessary because ResNet50 expects 3-channel input
            # (e.g., RGB images)
            model_input = model_input.repeat(1, 3, 1, 1)
        model_output = self.model(model_input)
        return model_output
