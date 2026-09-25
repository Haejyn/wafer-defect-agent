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


def one_hot(x_uint8: torch.Tensor) -> torch.Tensor:
    """(N,64,64) 값 0/1/2 → (N,3,64,64) float."""
    return F.one_hot(x_uint8.long(), 3).permute(0, 3, 1, 2).float()
