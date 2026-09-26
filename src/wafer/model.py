"""작은 CNN — 웨이퍼 맵(0/1/2)을 원-핫 3채널로 받는다. 전역 평균 풀링 뒤 256차원이 임베딩."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int = 9, width: int = 32):
        super().__init__()
        w = width
        self.b1, self.b2, self.b3, self.b4 = block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(8 * w, n_classes)
        self.emb_dim = 8 * w

    def features(self, x):
        """마지막 합성곱 특징맵(8×8) — 히트맵과 패치 단위 이상 점수에 쓴다."""
        x = F.max_pool2d(self.b1(x), 2)
        x = F.max_pool2d(self.b2(x), 2)
        x = F.max_pool2d(self.b3(x), 2)
        return self.b4(x)

    def embed(self, x):
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


class _Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(out)) + idt, inplace=True)


class ResNet18(nn.Module):
    """torchvision resnet18 과 같은 이름·구조(ImageNet 가중치를 그대로 불러온다). 64×64 입력이라 첫 max-pool 만 뺀다.
    torchvision 은 Python 3.14 용 빌드가 맞지 않아 직접 정의했다."""
    WEIGHTS = "https://download.pytorch.org/models/resnet18-f37072fd.pth"

    def __init__(self, n_classes: int = 9, pretrained: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        chans = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
        for i, (a, b, s) in enumerate(chans, 1):
            setattr(self, f"layer{i}", nn.Sequential(_Basic(a, b, s), _Basic(b, b, 1)))
        self.emb_dim = 512
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(512, n_classes)
        if pretrained:
            sd = torch.hub.load_state_dict_from_url(self.WEIGHTS, progress=False)
            missing, unexpected = self.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
            assert set(missing) == {"head.weight", "head.bias"} and not unexpected, (missing, unexpected)

    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def embed(self, x):
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


def build(arch: str, n_classes: int):
    return WaferCNN(n_classes) if arch == "cnn" else ResNet18(n_classes)


def one_hot(x_uint8: torch.Tensor) -> torch.Tensor:
    """(N,64,64) 값 0/1/2 → (N,3,64,64) float."""
    return F.one_hot(x_uint8.long(), 3).permute(0, 3, 1, 2).float()
