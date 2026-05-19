"""Training scaffold for the Task 2 MNIST digit classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNIST_DATA_DIR = TASK_ROOT / "data"


def download_mnist_dataset(data_dir: Path = DEFAULT_MNIST_DATA_DIR) -> Path:
    """Download torchvision MNIST into the Task 2 data directory."""
    import torchvision

    data_dir.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    return data_dir / "MNIST"


class MNISTClassifier(nn.Module):
    """Small PyTorch classifier scaffold for 28x28 MNIST crops."""

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        # TODO(student): fill in your custom model architectures
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)

        )

    def forward(self, inputs):
        return self.layers(inputs)
        raise NotImplementedError("MNIST classifier forward logic not implemented!")


def select_training_device(torch_module) -> str:
    # TODO(student): Pick the best accelerator available on the student's PC.
    # if torch reports CUDA is available:
    #     return "cuda" for NVIDIA GPU training
    # else if torch reports MPS is available:
    #     return "mps" for Apple Silicon GPU training
    # otherwise:
    #     return "cpu" so training still works without an accelerator
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"
    raise NotImplementedError("select_training_device is not implemented")


def train_mnist_classifier(dataset_dir: Path, output_path: Path) -> Path:
    
    from torch.utils.data import DataLoader, random_split
    import torchvision
    # TODO(student): Train the MNIST digit classifier used by model.py.
    # device = select_training_device(torch)
    # move the model and each batch to device
    # read training images and labels from dataset_dir
    # split examples into training and validation sets
    # preprocess every image the same way model.preprocess_mnist_crop does
    # model = MNISTClassifier()
    # choose loss function, optimizer, batch size, and number of epochs
    # train until validation accuracy is stable
    # save the trained model weights or serialized estimator to output_path
    # return output_path
    device = select_training_device(torch)
    print(f"Using device: {device}")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root=dataset_dir.parent, train=True, transform=transform, download=False)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = MNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    epochs = 10 
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad() 
            output = model(data)  
            loss = criterion(output, target) 
            loss.backward()      
            optimizer.step()  
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")


    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path
    raise NotImplementedError("MNIST training is not implemented")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2 MNIST digit classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_MNIST_DATA_DIR / "MNIST", help="Directory containing labeled MNIST board crops.")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "models" / "mnist_classifier.npz", help="Where to save the trained classifier.")
    parser.add_argument("--download-mnist", action="store_true", help="Download MNIST into tasks/task2-detector/data/MNIST before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_mnist:
        dataset_path = download_mnist_dataset(DEFAULT_MNIST_DATA_DIR)
        print(f"Downloaded MNIST dataset to: {dataset_path}")
        return

    output_path = train_mnist_classifier(args.dataset_dir, args.output)
    print(f"Saved MNIST classifier to: {output_path}")


if __name__ == "__main__":
    main()
