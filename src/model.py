import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import BertModel
from torchvision.models import resnet18


class MultimodalLightningModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super(MultimodalLightningModel, self).__init__()
        self.lr = lr

        # Text Encoder (BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

        # Image Encoder (ResNet)
        self.image_encoder = resnet18(pretrained=True)
        self.image_encoder.fc = nn.Linear(512, 256)  # Modify final layer

        # Fusion Layer
        self.fusion_layer = nn.Linear(768 + 256, 512)  # Combine text + image features

        # Classification Head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Binary output
        )
        self.criterion = nn.BCELoss()

    def forward(self, text_inputs, image_inputs):
        # Ensure text_inputs is a dictionary with expected keys
        text_features = self.text_encoder(
            input_ids=text_inputs['input_ids'].squeeze(1),
            attention_mask=text_inputs['attention_mask'].squeeze(1)
        ).pooler_output  # (Batch, 768)

        # Encode image
        image_features = self.image_encoder(image_inputs)  # (Batch, 256)

        # Fuse features
        combined_features = torch.cat((text_features, image_features), dim=1)  # (Batch, 1024)
        fused = self.fusion_layer(combined_features)

        # Classification
        output = self.classifier(fused)
        return output

    def training_step(self, batch, batch_idx):
        text, image, label = batch
        output = self(text, image).squeeze()
        loss = self.criterion(output, label.float())
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        text, image = batch
        output = self(text, image).squeeze()
        self.log('test_predictions', output)  # Log predictions for analysis

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
