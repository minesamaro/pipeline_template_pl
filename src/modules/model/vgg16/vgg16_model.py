from torchvision.models import vgg16
import torch


class VGG16Model(torch.nn.Module):
    def __init__(self, config):
        super(VGG16Model, self).__init__()
        self.model = vgg16(pretrained=False)
        self.model.classifier[6] = torch.nn.Linear(
            in_features=self.model.classifier[6].in_features,
            out_features=config.number_of_classes
        )

    def forward(self, model_input):
        # If the input has 1 channel, repeat it to 3 channels
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)
        model_output = self.model(model_input)
        return model_output
