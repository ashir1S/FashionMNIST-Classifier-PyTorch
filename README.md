# FashionMNIST-Classifier-PyTorch
FashionMNIST image classification using PyTorch, showcasing CNNs and model iteration.

**Project Overview**

This project demonstrates the development, training, and evaluation of three deep learning models for classifying fashion items from grayscale images in the FashionMNIST dataset. Implemented in PyTorch, the architectures explored are:

* **Baseline Linear Model**: A simple fully connected network
* **Non-linear Linear Model**: Adds ReLU activations for non-linearity
* **Convolutional Neural Network (CNN)**: Inspired by TinyVGG, optimized for image feature extraction

Best practices in data handling, model design, training loops, performance evaluation, visualization, and model persistence are emphasized throughout.

## Key Technologies & Tools

* **Framework**: PyTorch (`torch`, `torch.nn`, `torchvision`)
* **Data Pipeline**: `torch.utils.data.DataLoader` for batching and shuffling
* **Loss & Metrics**: `nn.CrossEntropyLoss`, custom accuracy functions, `torchmetrics`
* **Visualization**: `matplotlib`, `mlxtend` for confusion matrix
* **Utilities**: `tqdm` for progress bars, `pandas` for tabular summaries
* **Device Agnostic**: Automatic GPU/CPU selection via `torch.cuda.is_available()`

## Methodology

1. **Data Loading & Preprocessing**

   * Auto-download FashionMNIST with `torchvision.datasets`
   * Transform images to tensors and normalize
2. **Model Definitions**

   * `ModelV0`: Single linear layer
   * `ModelV1`: Two-layer MLP with ReLU
   * `ModelV2`: Multi-block CNN (conv → ReLU → pool)
3. **Training Loop**

   * Forward pass → loss computation → backpropagation → optimizer step
4. **Evaluation & Comparison**

   * Test loss, accuracy, and training time logged per model
   * Visualization of sample predictions
5. **Error Analysis**

   * Confusion matrix to highlight misclassifications
6. **Model Persistence**

   * Save and load best-performing model state (`.pth` files)

## Results Summary

| Model                   | Test Loss | Test Accuracy (%) | Train Time (s) |
| ----------------------- | --------- | ----------------- | -------------- |
| **FashionMNISTModelV0** | 0.4766    | 83.43             | 28.7           |
| **FashionMNISTModelV1** | 0.6850    | 75.02             | 31.5           |
| **FashionMNISTModelV2** | 0.3370    | 87.53             | 32.9           |

The CNN (`ModelV2`) outperforms linear counterparts in both accuracy and loss, demonstrating the power of convolutional architectures for image data.

## Setup & Usage

```bash
# Clone repository
git clone https://github.com/ashir1S/fashion-mnist-classifier.git
cd fashion-mnist-classifier

# Install dependencies
pip install torch torchvision matplotlib pandas tqdm torchmetrics mlxtend

# Run the notebook
jupyter notebook fashion_mnist_classifier.ipynb
```

The dataset is fetched automatically by the notebook via `torchvision.datasets.FashionMNIST`.

## Author

**Ashirwad Sinha**
Developed by [Ashirwad Sinha](https://github.com/ashir1S)  
Aspiring AI/ML Engineer | Passionate about PyTorch and Computer Vision

🧠 Built as part of hands-on deep learning exploration using the FashionMNIST dataset.
