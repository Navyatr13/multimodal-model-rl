import torch
import torch.optim as optim
import torch.nn.functional as F

class MultimodalRLTrainer:
    def __init__(self, model, dataloader, lr=1e-4):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.dataloader:
            text, image, labels = batch

            # Forward pass
            predictions = self.model(text, image).squeeze()

            # Compute rewards
            rewards = compute_reward((predictions > 0.5).long(), labels)

            # Compute loss (negative reward)
            loss = -(rewards * F.logsigmoid(predictions)).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)
