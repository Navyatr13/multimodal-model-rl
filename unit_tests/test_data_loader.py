# Unit Test for MultimodalDataset
import unittest
import pandas as pd
import os
import unittest
import pandas as pd
from PIL import Image  # Import this for image creation
from data_loader import MultimodalDataset  # Adjust the import based on your module structure

class TestMultimodalDataset(unittest.TestCase):

    def setUp(self):
        # Setup mock data and paths
        self.jsonl_file = "./data/mock_train.jsonl"
        self.image_dir = "./data/mock_images"
        self.save_path = "./data/mock_preprocessed.pkl"

        # Create mock JSONL file
        mock_data = [
            {"text": "Sample text 1", "img": "image1.jpg", "label": 0},
            {"text": "Sample text 2", "img": "image2.jpg", "label": 1},
        ]
        pd.DataFrame(mock_data).to_json(self.jsonl_file, lines=True, orient="records")

        # Create mock image files
        os.makedirs(self.image_dir, exist_ok=True)
        for img in ["image1.jpg", "image2.jpg"]:
            Image.new("RGB", (100, 100)).save(os.path.join(self.image_dir, img))

    def tearDown(self):
        # Clean up mock data
        os.remove(self.jsonl_file)
        for img in ["image1.jpg", "image2.jpg"]:
            os.remove(os.path.join(self.image_dir, img))
        os.rmdir(self.image_dir)
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def test_dataset_length(self):
        dataset = MultimodalDataset(
            jsonl_file=self.jsonl_file,
            image_dir=self.image_dir,
            save_path=self.save_path,
            load_processed=False
        )
        self.assertEqual(len(dataset), 2)

    def test_getitem(self):
        dataset = MultimodalDataset(
            jsonl_file=self.jsonl_file,
            image_dir=self.image_dir,
            save_path=self.save_path,
            load_processed=False
        )
        sample = dataset[0]
        self.assertIsNotNone(sample)
        self.assertEqual(sample[2], 0)  # Check label

    def test_save_and_load_processed(self):
        dataset = MultimodalDataset(
            jsonl_file=self.jsonl_file,
            image_dir=self.image_dir,
            save_path=self.save_path,
            load_processed=False
        )
        self.assertTrue(os.path.exists(self.save_path))

        # Load processed data
        loaded_dataset = MultimodalDataset(
            jsonl_file=self.jsonl_file,
            image_dir=self.image_dir,
            save_path=self.save_path,
            load_processed=True
        )
        self.assertEqual(len(loaded_dataset), 2)

if __name__ == "__main__":
    unittest.main()
