import torch

from src.model import CNNObjectDetector


def main():
    model = CNNObjectDetector(num_classes=80, num_anchors=3)
    image_batch = torch.randn(2, 3, 256, 256)
    predictions = model(image_batch)
    print(f"Predictions shape: {tuple(predictions.shape)}")


if __name__ == "__main__":
    main()
