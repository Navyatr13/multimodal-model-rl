from model import MultimodalLightningModel
from data_loader import MultimodalDataModule
from rl_trainer import RLTrainer

if __name__ == "__main__":
    # Load pretrained model
    model = MultimodalLightningModel.load_from_checkpoint('./checkpoints/your_best_model.ckpt')

    # Load DataModule
    data_module = MultimodalDataModule(
        train_file='./data/preprocessed_train.pkl',
        test_file='./data/preprocessed_test.pkl',
        batch_size=32
    )
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()

    # Train with RL
    rl_trainer = RLTrainer(model=model, dataloader=train_loader, lr=1e-4)
    num_epochs = 5
    for epoch in range(num_epochs):
        avg_loss = rl_trainer.train_one_epoch()
        print(f"Epoch {epoch + 1}, RL Loss: {avg_loss}")

    # Save the RL fine-tuned model
    torch.save(model.state_dict(), './checkpoints/rl_finetuned_model.pth')
