import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_mobilenet_model(num_classes):
    # Load pretrained MobileNetV2
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # 🔒 Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 🔄 Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model
