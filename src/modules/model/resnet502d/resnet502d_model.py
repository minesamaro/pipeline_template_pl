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
        model_output = self.model(model_input.repeat(1, 3, 1, 1))
        return model_output
