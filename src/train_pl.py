import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import MultimodalDataModule
from model import MultimodalLightningModel

# Define paths to your pickled datasets
train_file = './data/preprocessed_train.pkl'
val_file = './data/preprocessed_test.pkl'
image_dir = './data/'
# Initialize DataModule and Model
data_module = MultimodalDataModule(train_file, val_file, batch_size=32)
model = MultimodalLightningModel(lr=1e-4)

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3)
checkpoint = ModelCheckpoint(dirpath='./checkpoints', save_top_k=1, monitor='val_loss', mode='min')

trainer = pl.Trainer(
    max_epochs=10,
    devices =1 if torch.cuda.is_available() else 0,
    callbacks=[early_stop, checkpoint]
)
# Start training
data_module.setup("fit")
train_loader = data_module.train_dataloader()
trainer.fit(model, data_module)

# Access the test dataloader
data_module.setup("test")  # For testing stage
test_loader = data_module.test_dataloader()
trainer.test(model, datamodule=data_module)


