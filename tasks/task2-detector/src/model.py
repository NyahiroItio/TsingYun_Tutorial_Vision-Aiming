"""MNIST digit model scaffold for Task 2.

Detector code should call the inference function in this module. Training code
lives in train.py so detector.py stays focused on board detection, corner
geometry, and PnP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import torch
import torch.nn.functional as F

RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray

from train import MNISTClassifier

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_classifier.npz"


def preprocess_mnist_crop(board_crop: ImageLike) -> np.ndarray:
    # TODO(student): Convert a detected board crop into classifier input.
    # convert board_crop to a single-channel image using cv2
    # resize the grayscale crop to 28x28, for example with cv2.resize(...)
    # normalize values to [0, 1]
    # convert the result to the tensor/array shape expected by your classifier
    # return normalized input array
    board_crop = np.array(board_crop, dtype=np.uint8)
    gray = cv2.cvtColor(board_crop, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.1307) / 0.3081

    return normalized[np.newaxis, np.newaxis, :, :]
    raise NotImplementedError("preprocess_mnist_crop is not implemented")


def load_mnist_model(model_path: Path = DEFAULT_MODEL_PATH) -> object:
    # TODO(student): Load your trained MNIST classifier from disk.
    # Input: model_path.
    # Output: a trained classifier model ready for inference.
    # If you use PyTorch, instantiate the model, load the weights, and switch to eval mode.
    model = MNISTClassifier()

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
    raise NotImplementedError("load_mnist_model is not implemented")


def predict_mnist_digit(model: object, model_input: np.ndarray) -> tuple[int, float]:
    # TODO(student): Run classifier inference and convert scores to digit/confidence.
    # convert model_input to a torch tensor if needed
    # run inference under torch.no_grad()
    # apply F.softmax(...) if the model returns logits
    # digit = argmax(probabilities)
    # confidence = probabilities[digit]
    # return digit, confidence
    img_tensor = torch.from_numpy(model_input)
    
    with torch.no_grad(): 
        logits = model(img_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probabilities, 1)
        
    return int(pred_class.item()), float(confidence.item())
    raise NotImplementedError("predict_mnist_digit is not implemented")


def classify_mnist_digit(board_crop: ImageLike, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[int, float]:
    # TODO(student): Classify the MNIST digit shown on one detected board crop.
    # model_input = preprocess_mnist_crop(board_crop)
    # model = load_mnist_model(model_path)
    # digit, confidence = predict_mnist_digit(model, model_input)
    # return digit, confidence
    model_input = preprocess_mnist_crop(board_crop)
    model = load_mnist_model(model_path)
    digit, confidence = predict_mnist_digit(model, model_input)
    return digit, confidence
    raise NotImplementedError("classify_mnist_digit is not implemented")
