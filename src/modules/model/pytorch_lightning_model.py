from src.modules.model.efficient_net.pytorch_lightning_efficient_net_model \
    import PyTorchLightningEfficientNetModel
from src.modules.model.resnet502d.pytorch_lightning_resnet50_2d_model \
    import PyTorchLightningResNet502dModel
from src.modules.model.vgg16.pytorch_lightning_vgg16_model \
    import PyTorchLightningVGG16Model
from src.modules.model.resnet503d.pytorch_lightning_resnet50_3d_model \
    import PyTorchLightningResNet503dModel
from src.modules.model.vgg163d.pytorch_lightning_vgg163d_model \
    import PyTorchLightningVGG163dModel

class PyTorchLightningModel:
    def __new__(cls, config, experiment_execution_paths):
        if config.model_name == "EfficientNet":
            return PyTorchLightningEfficientNetModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "ResNet502d":
            return PyTorchLightningResNet502dModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "VGG16":
            return PyTorchLightningVGG16Model(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "ResNet503d":
            return PyTorchLightningResNet503dModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "VGG163d":
            return PyTorchLightningVGG163dModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        else:
            raise ValueError(
                f"Invalid model name: {config.model_name}. "
                f"Supported datasets are 'EfficientNet'."
            )