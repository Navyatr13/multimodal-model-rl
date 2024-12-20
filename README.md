
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

## Installation

### 1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multimodal-model-rl.git
   cd multimodal-model-rl

### 2. Create a virtual environment:
    ```bash
    conda create -n mm_env
    conda activate mm_env
   
### 3. Install dependencies:
    ```
   pip install -r requirements.txt
   ```
### 4. Dataset Preparation
Place your dataset files (train.jsonl, test.jsonl, images) in the data/ directory.
Preprocess the dataset: Run the preprocessing script in data_loader.py to generate pickle files:
``` 
python src/data_loader.py
```
### 5. Usage
##### Training the Model
To train the model:
``` 
python src/train_pl.py
```
##### Testing the Model
To evaluate the model:
```
python src/train_pl.py
```

### 6. Unit Testing
Run unit tests to verify functionality:
``` python -m unittest discover -s unit_tests```
### Model Architecture
The model combines textual and visual features using:

Text Encoder: Pretrained BERT from the transformers library.
Image Encoder: Pretrained ResNet18 from torchvision.models.
Fusion Layer: Linear layer combining outputs from the text and image encoders.
Classifier: Binary classification head.
Requirements
Python 3.8+
PyTorch
PyTorch Lightning
Transformers
torchvision
pandas
Pillow
tqdm
Acknowledgments
PyTorch Lightning for simplifying model training.
Hugging Face Transformers for pretrained BERT.
torchvision for pretrained ResNet.
License
This project is licensed under the MIT License. See the LICENSE file for details.
