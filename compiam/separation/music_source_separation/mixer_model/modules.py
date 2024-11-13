import torch
import torch.nn as nn


# Time-Frequency Modulation (directly ported from original code by KUIELab/TFC-TDF)


class TFC(nn.Module):
    def __init__(self, c, l, k):
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=c,
                        out_channels=c,
                        kernel_size=k,
                        stride=1,
                        padding=k // 2,
                    ),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x


# Dense TFC Block (directly ported from original code by KUIELab/TFC-TDF)


class DenseTFC(nn.Module):
    def __init__(self, c, l, k):
        super(DenseTFC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):

            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=c,
                        out_channels=c,
                        kernel_size=k,
                        stride=1,
                        padding=k // 2,
                    ),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            out = layer(x)
            x = torch.cat([out, x], 1)
        return self.conv[-1](x)


# TFC TDF module (directly ported from original code by KUIELab/TFC-TDF)


class TFC_TDF(nn.Module):
    def __init__(self, c, l, f, k, bn, dense=False, bias=True):
        super(TFC_TDF, self).__init__()

        self.use_tdf = bn is not None
        self.tfc = DenseTFC(c, l, k) if dense else TFC(c, l, k)
        self.bn = bn

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias), nn.BatchNorm2d(c), nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )

    def forward(self, x):
        out = self.tdf(x)
        x = self.tfc(x)
        return x + out if self.use_tdf else x
