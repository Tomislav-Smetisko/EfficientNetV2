# EfficientNetV2 Face Recognition

This project contains an implementation of the EfficientNetV2-S neural network for face identification using the Labeled Faces in the Wild (LFW) dataset.

The model is implemented in PyTorch and performs closed-set face identification by predicting the most likely identity from the classes used during training.

Training uses images resized to 160 × 160 pixels, the Adam optimizer, and CrossEntropyLoss. The repository also includes a testing script for identifying a person from a single image and displaying the Top-K predictions.
