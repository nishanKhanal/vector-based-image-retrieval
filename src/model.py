import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetTransferModel(nn.Module):
    def __init__(self, num_classes=101, embedding_size=128, base_model='resnet18', pretrained=True):
        super(ResNetTransferModel, self).__init__()

        # Load selected ResNet base model
        self.resnet = getattr(models, base_model)(pretrained=pretrained)

        # Freeze all layers except the last layer(layer4) and fc layer
        for name, param in self.resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Dynamically grab fc input size 
        self.resnet_fc_in_features = self.resnet.fc.in_features  
        
        self.resnet.fc = nn.Identity()  # Remove the final FC layer
        

        # Add our custom classifier with dropout
        self.embedding = nn.Sequential(
            nn.Linear(self.resnet_fc_in_features , embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )

        # Classification layer
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # Extract features from ResNet
        features = self.resnet(x)
        # Get embedding
        embedding = self.embedding(features)
        # Get class predictions
        logits = self.classifier(embedding)
        return logits

    def extract_features(self, x):
        """Extract feature embeddings for image retrieval"""
        features = self.resnet(x)
        embedding = self.embedding(features)
        # Normalize embedding to unit length for better similarity search
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding