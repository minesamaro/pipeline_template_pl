from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch
import timm
import torchvision.transforms as transforms



class ViTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Load pre-trained ViT model
        model_name = config.get('model_name', 'vit_base_patch16_224')
        pretrained = config.get('pretrained', True)
        num_classes = config.get('num_classes', 1)  # 1 for binary classification
        
        img_size = 224  # Default to 224
        self.preprocess = transforms.Resize((img_size, img_size))
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size  # This allows different input sizes
        )
        
        # If you want to freeze the backbone and only train the classifier
        if config.get('freeze_backbone', True):
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the head
            for param in self.backbone.head.parameters():
                param.requires_grad = True    
    def forward(self, x, interpolate_pos_encoding=True):
        if x.shape[1] == 1:
            # If the input has only one channel, repeat it to make it 3 channels
            # This is necessary because ResNet50 expects 3-channel input
            # (e.g., RGB images)
            x = x.repeat(1, 3, 1, 1)
        x = self.preprocess(x)
        return self.backbone(x)
