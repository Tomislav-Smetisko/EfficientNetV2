import os
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


DATASET_DIR = "lfw_dataset"

IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

MIN_IMAGES_PER_CLASS = 20

TEST_SIZE = 0.2

MODEL_SAVE_PATH = "efficientnetv2_lfw_identification.pth"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model):
    total = 0

    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()

    return total


def get_model_size_mb(path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


class ConvBNAct(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        activation=True
    ):
        padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        ]

        if activation:
            layers.append(nn.SiLU(inplace=True))

        super().__init__(*layers)


class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()

        reduced_channels = max(1, int(in_channels * se_ratio))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(
            in_channels,
            reduced_channels,
            kernel_size=1
        )

        self.act = nn.SiLU(inplace=True)

        self.fc2 = nn.Conv2d(
            reduced_channels,
            in_channels,
            kernel_size=1
        )

        self.scale = nn.Sigmoid()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.scale(scale)

        return x * scale


class StochasticDepth(nn.Module):

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0.0:
            return x

        keep_prob = 1.0 - self.drop_rate

        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device
        )

        binary_tensor = torch.floor(random_tensor)

        return x / keep_prob * binary_tensor


class FusedMBConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        drop_rate=0.0
    ):
        super().__init__()

        assert stride in [1, 2]

        hidden_channels = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        if expand_ratio != 1:
            layers.append(
                ConvBNAct(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=1,
                    activation=True
                )
            )

            layers.append(
                ConvBNAct(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    activation=False
                )
            )

        else:
            layers.append(
                ConvBNAct(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=1,
                    activation=True
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(drop_rate)

    def forward(self, x):
        result = self.block(x)

        if self.use_residual:
            result = self.stochastic_depth(result)
            result = result + x

        return result


class MBConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        se_ratio=0.25,
        drop_rate=0.0
    ):
        super().__init__()

        assert stride in [1, 2]

        hidden_channels = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        if expand_ratio != 1:
            layers.append(
                ConvBNAct(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    activation=True
                )
            )

        layers.append(
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_channels,
                activation=True
            )
        )

        layers.append(
            SqueezeExcitation(
                in_channels=hidden_channels,
                se_ratio=se_ratio
            )
        )

        layers.append(
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                activation=False
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(drop_rate)

    def forward(self, x):
        result = self.block(x)

        if self.use_residual:
            result = self.stochastic_depth(result)
            result = result + x

        return result


class EfficientNetV2S(nn.Module):

    def __init__(
        self,
        num_classes,
        dropout_rate=0.2,
        stochastic_depth_rate=0.2
    ):
        super().__init__()

        # block_type, repeats, kernel_size, stride, expand_ratio, in_channels, out_channels, se_ratio
        config = [
            ["fused", 2, 3, 1, 1, 24, 24, 0.0],
            ["fused", 4, 3, 2, 4, 24, 48, 0.0],
            ["fused", 4, 3, 2, 4, 48, 64, 0.0],
            ["mbconv", 6, 3, 2, 4, 64, 128, 0.25],
            ["mbconv", 9, 3, 1, 6, 128, 160, 0.25],
            ["mbconv", 15, 3, 2, 6, 160, 256, 0.25],
        ]

        total_blocks = sum(row[1] for row in config)
        block_index = 0

        layers = []

        layers.append(
            ConvBNAct(
                in_channels=3,
                out_channels=24,
                kernel_size=3,
                stride=2,
                groups=1,
                activation=True
            )
        )

        for block_type, repeats, kernel_size, stride, expand_ratio, in_channels, out_channels, se_ratio in config:
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                block_in_channels = in_channels if i == 0 else out_channels

                drop_rate = stochastic_depth_rate * block_index / total_blocks

                if block_type == "fused":
                    layers.append(
                        FusedMBConv(
                            in_channels=block_in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=block_stride,
                            expand_ratio=expand_ratio,
                            drop_rate=drop_rate
                        )
                    )

                else:
                    layers.append(
                        MBConv(
                            in_channels=block_in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=block_stride,
                            expand_ratio=expand_ratio,
                            se_ratio=se_ratio,
                            drop_rate=drop_rate
                        )
                    )

                block_index += 1

        self.features = nn.Sequential(*layers)

        self.head = ConvBNAct(
            in_channels=256,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            groups=1,
            activation=True
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class LFWDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def collect_images_by_person(dataset_dir: str):

    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset ne postoji: {dataset_dir}")

    images_by_person = defaultdict(list)

    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                images_by_person[person_name].append(str(image_path))

    return images_by_person


def prepare_train_test_split(dataset_dir: str):

    images_by_person = collect_images_by_person(dataset_dir)

    selected_people = {
        person: images
        for person, images in images_by_person.items()
        if len(images) >= MIN_IMAGES_PER_CLASS
    }

    if len(selected_people) == 0:
        raise ValueError(
            "Nema osoba s dovoljnim brojem slika. "
        )

    person_names = sorted(selected_people.keys())

    person_to_class_index = {
        person_name: index
        for index, person_name in enumerate(person_names)
    }

    class_index_to_person = {
        index: person_name
        for person_name, index in person_to_class_index.items()
    }

    train_samples = []
    test_samples = []

    for person_name, image_paths in selected_people.items():
        label = person_to_class_index[person_name]

        train_paths, test_paths = train_test_split(
            image_paths,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )

        for path in train_paths:
            train_samples.append((path, label))

        for path in test_paths:
            test_samples.append((path, label))

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    print(f"Ukupan broj identiteta u originalnom LFW skupu: {len(images_by_person)}")
    print(f"Broj korištenih identiteta: {len(selected_people)}")
    print(f"Broj trening slika: {len(train_samples)}")
    print(f"Broj testnih slika: {len(test_samples)}")

    return train_samples, test_samples, person_to_class_index, class_index_to_person


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(all_labels)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    total_inference_time = 0.0
    total_images = 0

    for images, labels in tqdm(test_loader, desc="Evaluacija"):
        images = images.to(device)
        labels = labels.to(device)

        sync_device(device)
        start_time = time.perf_counter()

        outputs = model(images)

        sync_device(device)
        end_time = time.perf_counter()

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        total_inference_time += end_time - start_time
        total_images += images.size(0)

    accuracy = accuracy_score(all_labels, all_predictions)

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    average_inference_time = total_inference_time / total_images

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "average_inference_time": average_inference_time,
        "all_labels": all_labels,
        "all_predictions": all_predictions
    }

    return metrics


def main():
    set_seed(RANDOM_SEED)

    device = get_device()

    train_samples, test_samples, person_to_class_index, class_index_to_person = prepare_train_test_split(DATASET_DIR)

    train_transform, test_transform = get_transforms()

    train_dataset = LFWDataset(
        samples=train_samples,
        transform=train_transform
    )

    test_dataset = LFWDataset(
        samples=test_samples,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    num_classes = len(person_to_class_index)

    model = EfficientNetV2S(
        num_classes=num_classes,
        dropout_rate=0.1,
        stochastic_depth_rate=0.0
    )

    model.to(device)

    print(f"Broj klasa: {num_classes}")
    print(f"Broj parametara: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    sync_device(device)
    training_start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f}"
        )

    sync_device(device)
    training_end_time = time.perf_counter()

    total_training_time = training_end_time - training_start_time

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "person_to_class_index": person_to_class_index,
            "class_index_to_person": class_index_to_person,
            "image_size": IMAGE_SIZE,
            "num_classes": num_classes
        },
        MODEL_SAVE_PATH
    )

    model_size_mb = get_model_size_mb(MODEL_SAVE_PATH)

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )

    print("\n================ REZULTATI ================")
    print(f"Accuracy / Top-1 accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro precision:                 {metrics['macro_precision']:.4f}")
    print(f"Macro recall:                    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:                        {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:                     {metrics['weighted_f1']:.4f}")
    print(f"Vrijeme treniranja:              {total_training_time:.2f} s")
    print(f"Prosječno vrijeme po slici:      {metrics['average_inference_time'] * 1000:.4f} ms")
    print(f"Broj parametara:                 {count_parameters(model):,}")
    print(f"Veličina spremljenog modela:     {model_size_mb:.2f} MB")
    print(f"Model spremljen kao:             {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()