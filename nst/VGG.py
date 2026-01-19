import torch.nn as nn
import torchvision.models as models

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1",
        }

        self.model = vgg[:29]

        for p in self.model.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        features = {}

        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        
        return features