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
