import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


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

        config = [
            # block_type, repeats, kernel_size, stride, expand_ratio, in_channels, out_channels, se_ratio
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

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


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


def load_model(model_path, device):
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    return model


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    return image


@torch.no_grad()
def predict_identity(model, image_path, transform, class_index_to_person, device, top_k=5):
    model.eval()

    image = load_image(image_path, transform)
    image = image.to(device)

    sync_device(device)
    start_time = time.perf_counter()

    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)

    sync_device(device)
    end_time = time.perf_counter()

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1
    )

    top_probabilities = top_probabilities.squeeze(0).cpu().numpy()
    top_indices = top_indices.squeeze(0).cpu().numpy()

    results = []

    for index, probability in zip(top_indices, top_probabilities):
        if index in class_index_to_person:
            person_name = class_index_to_person[index]
        else:
            person_name = class_index_to_person[str(index)]

        results.append(
            {
                "person": person_name,
                "probability": float(probability)
            }
        )

    inference_time = end_time - start_time

    return results, inference_time


def identify_image(model_path_str, image_path_str, top_k=5):
    model_path = Path(model_path_str)
    image_path = Path(image_path_str)

    if not model_path.exists():
        raise FileNotFoundError(f"Model ne postoji: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Slika ne postoji: {image_path}")

    device = get_device()
    checkpoint = load_model(model_path, device)

    num_classes = checkpoint["num_classes"]
    image_size = checkpoint["image_size"]
    class_index_to_person = checkpoint["class_index_to_person"]

    model = EfficientNetV2S(
        num_classes=num_classes,
        dropout_rate=0.1,
        stochastic_depth_rate=0.0
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_transform(image_size)

    results, inference_time = predict_identity(
        model=model,
        image_path=image_path,
        transform=transform,
        class_index_to_person=class_index_to_person,
        device=device,
        top_k=top_k
    )

    top_results = []

    for result in results:
        top_results.append(
            {
                "name": result["person"],
                "probability": round(result["probability"] * 100, 2)
            }
        )

    return {
        "model": "EfficientNetV2",
        "top_results": top_results,
        "inference_time": round(inference_time * 1000, 4)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Identifikacija osobe pomoću EfficientNetV2 modela."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetv2_lfw_identification.pth",
        help="Putanja do spremljenog .pth modela."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Putanja do slike osobe koja se identificira."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Broj najvjerojatnijih identiteta za prikaz."
    )

    args = parser.parse_args()

    results = identify_image(args.model, args.image, args.top_k)

    print("\n================ REZULTAT IDENTIFIKACIJE ================")
    print(f"Vrijeme zaključivanja: {results['inference_time']} ms")

    best_result = results["top_results"][0]

    print("\nNajvjerojatniji identitet:")
    print(f"{best_result['name']} ({best_result['probability']}%)")

    print(f"\nTop {args.top_k} predikcija:")

    for i, result in enumerate(results["top_results"], start=1):
        print(
            f"{i}. {result['name']} "
            f"- {result['probability']}%"
        )


if __name__ == "__main__":
    main()