import torch
from model import MultimodalLightningModel
from data_loader import MultimodalDataModule
from sklearn.metrics import accuracy_score

def test_rl_model(model_path, data_module):
    # Load RL-trained model
    model = MultimodalLightningModel.load_from_checkpoint(model_path)
    model.eval()

    # Prepare test data
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    predictions = []
    ground_truths = []

    # Run predictions
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                text, images, labels = batch  # If test data includes labels
                ground_truths.extend(labels.cpu().numpy())
            else:
                text, images = batch

            outputs = model(text, images).squeeze()
            preds = (outputs > 0.5).long()  # Binary classification
            predictions.extend(preds.cpu().numpy())

    # Compute metrics if labels are available
    if ground_truths:
        acc = accuracy_score(ground_truths, predictions)
        print(f"Test Accuracy: {acc * 100:.2f}%")

    # Return predictions
    return predictions

if __name__ == "__main__":
    model_path = "./checkpoints/rl_finetuned_model.pth"
    train_file = './data/preprocessed_train.pkl'
    test_file = './data/preprocessed_test.pkl'
    batch_size = 32

    # Initialize DataModule
    data_module = MultimodalDataModule(
        train_file=train_file,
        test_file=test_file,
        batch_size=batch_size
    )

    # Test RL Model
    test_rl_model(model_path, data_module)
