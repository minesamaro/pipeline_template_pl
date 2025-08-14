from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch
import timm


class ViTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Load pre-trained ViT model
        model_name = config.get('model_name', 'vit_base_patch16_224')
        pretrained = config.get('pretrained', True)
        num_classes = config.get('num_classes', 1)  # 1 for binary classification
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # If you want to freeze the backbone and only train the classifier
        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the head
            for param in self.backbone.head.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)
