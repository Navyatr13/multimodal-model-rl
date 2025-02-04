# Multimodal Model with PyTorch Lightning

This project implements a multimodal machine learning model combining text and image inputs using PyTorch Lightning. The model is trained and tested on a dataset designed for binary classification tasks.

## Features
- Uses a BERT-based text encoder for textual features.
- Uses a ResNet-based image encoder for visual features.
- Combines text and image features through a fusion layer.
- Binary classification output with a Sigmoid activation.
- Supports efficient training with PyTorch Lightning.
- Modular data handling with `MultimodalDataModule`.

---

## Project Structure
├── data/ # Data directory │ ├── train.jsonl # Training data (JSONL format) │ ├── test.jsonl # Test data (JSONL format) │ ├── preprocessed_train.pkl # Preprocessed training dataset │ ├── preprocessed_test.pkl # Preprocessed test dataset ├── src/ # Source code directory │ ├── data_loader.py # Dataset and DataLoader implementation │ ├── model.py # Multimodal Lightning Model definition │ ├── train_pl.py # Training and testing pipeline ├── unit_tests/ # Unit test directory │ ├── test_data_loader.py # Tests for data loading │ ├── test_training.py # Tests for training pipeline └── README.md # Project README

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Navyatr13/multimodal-model-rl.git
   cd multimodal-model-rl
2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
   
### Dataset Preparation
1. Place your dataset files (train.jsonl, test.jsonl, images) in the data/ directory.

2. Preprocess the dataset: Run the preprocessing script in data_loader.py to generate pickle files:
    ```bash

### Usage
#### Training the Model
To train the model:

    ```bash
    python src/train_pl.py
#### Testing the Model
To evaluate the model:

    ```bash
    python src/train_pl.py

### Unit Testing
Run unit tests to verify functionality:
```python -m unittest discover -s unit_tests```

### Model Architecture

The model combines textual and visual features using:

- **Text Encoder**: Pretrained BERT from the `transformers` library.
- **Image Encoder**: Pretrained ResNet18 from `torchvision.models`.
- **Fusion Layer**: Linear layer combining outputs from the text and image encoders.
- **Classifier**: Binary classification head for binary output.

---

### Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Transformers
- torchvision
- pandas
- Pillow
- tqdm

---

### Acknowledgments

- PyTorch Lightning for simplifying model training.
- Hugging Face Transformers for pretrained BERT.
- torchvision for pretrained ResNet.

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
