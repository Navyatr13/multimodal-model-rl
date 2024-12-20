import unittest
import torch
import pytorch_lightning as pl
from model import MultimodalLightningModel
from data_loader import MultimodalDataModule
from torch.utils.data import DataLoader, Dataset

# Mock Dataset for Testing
class MockDataset(Dataset):
    def __init__(self, num_samples=10, has_labels=True):
        self.num_samples = num_samples
        self.has_labels = has_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
        }
        image = torch.rand(3, 224, 224)  # Example RGB image
        if self.has_labels:
            label = torch.tensor(1.0)  # Binary classification label
            return text, image, label
        else:
            return text, image

# Test Class
class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        # Initialize the model
        self.model = MultimodalLightningModel(lr=1e-4)

        # Initialize the data module with mock datasets
        self.data_module = MultimodalDataModule(
            train_file=None,
            test_file=None,
            batch_size=4
        )

        # Mock train and test datasets
        self.data_module.train_dataset = MockDataset(num_samples=20, has_labels=True)
        self.data_module.test_dataset = MockDataset(num_samples=10, has_labels=False)

    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, MultimodalLightningModel)

    def test_dataloader(self):
        """Test if dataloaders provide batches correctly."""
        train_loader = self.data_module.train_dataloader()
        for batch in train_loader:
            self.assertEqual(len(batch), 3)  # Expecting text, image, label
            text, image, label = batch
            self.assertEqual(text["input_ids"].shape, (4, 128))  # Batch size 4
            self.assertEqual(image.shape, (4, 3, 224, 224))  # Batch size 4
            self.assertEqual(label.shape, (4,))

    def test_training_step(self):
        """Test a single training step."""
        train_loader = self.data_module.train_dataloader()
        batch = next(iter(train_loader))
        loss = self.model.training_step(batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_training_pipeline(self):
        """Test the full training pipeline."""
        trainer = pl.Trainer(
            max_epochs=1,
            gpus=0,  # Set to 1 if using a GPU
            fast_dev_run=True  # Run quickly for debugging
        )
        trainer.fit(self.model, datamodule=self.data_module)

    def test_testing_pipeline(self):
        """Test the testing pipeline."""
        trainer = pl.Trainer(gpus=0, fast_dev_run=True)
        trainer.test(self.model, datamodule=self.data_module)

if __name__ == "__main__":
    unittest.main()
