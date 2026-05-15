import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import DistilBertModel


class PhishingMultimodalNet(nn.Module):
    def __init__(self):
        super(PhishingMultimodalNet, self).__init__()

        # Obraz
        # ResNet18
        self.vision = resnet18(weights=ResNet18_Weights.DEFAULT)
        #tylko wektor cech
        self.vision.fc = nn.Identity()

        # Tekst
        # DistilBERT
        self.text = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, images, input_ids, attention_mask):

        vis_features = self.vision(images)  # [batch_size, 512]


        text_outputs = self.text(input_ids=input_ids, attention_mask=attention_mask)

        text_features = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]


        combined_features = torch.cat((vis_features, text_features), dim=1)  # [batch_size, 1280]


        output = self.classifier(combined_features)
        return output