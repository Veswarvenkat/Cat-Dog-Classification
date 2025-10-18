# Cat & Dog Image Classification (CNN)

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images as either "Cat" or "Dog". It's a classic binary image classification problem that demonstrates how to load and process a custom image dataset.

## Project Overview

The notebook covers the complete workflow for a basic computer vision task:
* **Data Loading:** Loads the "PetImages" dataset from a Google Drive folder.
* **Data Pipeline:** Uses `image_dataset_from_directory` to create efficient training and validation data pipelines.
* **Data Subsetting:** Intentionally uses `.take()` to select only a small number of batches for a very fast demonstration run.
* **Model Building:** Defines a simple 3-layer CNN.
* **Training:** Trains the model for 5 epochs.
* **Visualization:** Plots the training vs. validation accuracy over the epochs.

---

## Dataset

This project is designed to work with the "PetImages" (Cats vs. Dogs) dataset, which is often sourced from Kaggle.

**Important:** The code assumes the dataset is located in your Google Drive and expects the following directory structure:

The notebook will mount Google Colab to `/content/drive` and set the `dataset_dir` to `/content/drive/MyDrive/PetImages`.

### Note on Data Size
This notebook is configured for a **quick demonstration**, not for full-scale training. The following lines limit the dataset to a tiny fraction of the total images:

* `train_ds = ... .take(10)` (Uses only 10 batches for training)
* `validation_ds = ... .take(5)` (Uses only 5 batches for validation)

To train on the **entire dataset** for better accuracy, you must **remove** the `.take()` methods from both `train_ds` and `validation_ds`.

---

## Model Architecture

The model is a `Sequential` stack of three convolutional blocks:

1.  **Conv Block 1:** `Conv2D` (32 filters, 3x3 kernel, `relu`) + `MaxPooling2D`
2.  **Conv Block 2:** `Conv2D` (64 filters, 3x3 kernel, `relu`) + `MaxPooling2D`
3.  **Conv Block 3:** `Conv2D` (128 filters, 3x3 kernel, `relu`) + `MaxPooling2D`
4.  **Classifier Head:**
    * `Flatten`: Converts the 2D feature maps into a 1D vector.
    * `Dense` (128 units, `relu` activation)
    * `Dense` (1 unit, `sigmoid` activation): The output layer, producing a single value between 0 (Cat) and 1 (Dog).

The model is compiled with:
* **Optimizer:** `adam`
* **Loss Function:** `binary_crossentropy` (perfect for 0/1 binary classification)
* **Metrics:** `accuracy`

---

## Requirements

* `tensorflow`
* `matplotlib`
* `google.colab` (if running in Google Colab)

You can install the main dependencies using pip:
```bash
pip install tensorflow matplotlib
