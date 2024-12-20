import pandas as pd
import os
import pickle
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, batch_size=32):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Load preprocessed training dataset from pickle
            with open(self.train_file, 'rb') as f:
                self.train_dataset = pickle.load(f)

        if stage == "test" or stage is None:
            # Load preprocessed test dataset from pickle
            with open(self.test_file, 'rb') as f:
                self.test_dataset = pickle.load(f)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=0,
                          batch_size=self.batch_size,
                          shuffle=True,
                          persistent_workers=True
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=0,
                          batch_size=self.batch_size,
                          shuffle=False)


class MultimodalDataset(Dataset):
    def __init__(self, jsonl_file, image_dir, save_path=None, load_processed=False, has_labels=True):
        print(jsonl_file,"jsonl_file**************************")
        self.has_labels = has_labels
        if load_processed and save_path and os.path.exists(save_path):
            # Load pre-processed data
            with open(save_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            # Load data from JSONL file
            self.data = pd.read_json(jsonl_file, lines=True)
            self.image_dir = image_dir

            def file_exists(img):
                img_path = os.path.join(self.image_dir, img)
                if not os.path.isfile(img_path):
                    print(f"Missing file: {img_path}")
                    return False
                return True

            self.data = self.data[self.data['img'].apply(file_exists)].reset_index(drop=True)

            # Initialize tokenizer and image transformations
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Preprocess and save data
            def process_row(row):
                try:
                    # Process image
                    img_path = os.path.join(self.image_dir, row['img'])
                    image = Image.open(img_path).convert("RGB")
                    preprocessed_image = self.transform(image)

                    # Process text
                    tokenized_text = self.tokenizer(
                        row['text'],
                        max_length=128,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    if self.has_labels:
                        return tokenized_text, preprocessed_image, row['label']
                    else:
                        return tokenized_text, preprocessed_image
                except Exception as e:
                    print(f"Error processing row {row['img']}: {e}")
                    return None

            self.data = self.data.apply(process_row, axis=1).dropna()

            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

if __name__ == "__main__":
    train_dataset = MultimodalDataset(
        jsonl_file="./data/train.jsonl",
        image_dir="./data/",
        save_path="./data/preprocessed_train.pkl",
        load_processed= True,  # Set to True after initial run
        has_labels=True
    )
    test_dataset = MultimodalDataset(
        jsonl_file="./data/test.jsonl",
        image_dir="./data/",
        save_path="./data/preprocessed_test.pkl",
        load_processed=False,
        has_labels=False
    )

    print(f"Number of samples in training dataset: {len(train_dataset)}")
    print(f"Number of samples in test dataset: {len(test_dataset)}")

    sample = train_dataset[0]
    if sample:
        print(f"Sample tokenized text: {sample[0]}")
        print(f"Sample image shape: {sample[1].shape}")
        print(f"Sample label: {sample[2]}")

    test_sample = test_dataset[0]
    if test_sample:
        print(f"Test sample tokenized text: {test_sample[0]}")
        print(f"Test sample image shape: {test_sample[1].shape}")

