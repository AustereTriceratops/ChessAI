import torch



class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size

        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(channels)

        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)

        self.pad = torch.nn.ZeroPad2d((2, 2, 2, 2))

    def forward(self, x):
        x_ = self.bn(x)
        x_ = self.activation(self.pad(x_))
        x_ = self.conv1(x_)

        x_ = self.bn(x_)
        x_ = self.activation(self.pad(x_))
        x_ = self.conv2(x_)

        return x + x_

class ResNet(torch.nn.Module):
    def __init__(self, channels, depth=1, kernel_size=5):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.depth = depth

        self.layers = torch.nn.ModuleList(
            [ResBlock(self.channels, kernel_size=self.kernel_size)]
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class ChessNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(48)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.flatten = torch.nn.Flatten()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

        self.pad = torch.nn.ZeroPad2d((2, 2, 2, 2))
        self.upsample = torch.nn.Conv2d(in_channels=20, out_channels=48, kernel_size=5)
        self.resnet = ResNet(48, depth=2, kernel_size=5)
        self.downsample = torch.nn.Conv2d(in_channels=48, out_channels=8, kernel_size=5)
        self.value_head = torch.nn.Linear(512, 1)

        self.policy_head = torch.nn.Conv2d(in_channels=48, out_channels=73, kernel_size=5)

    def forward(self, x):
        x = self.pad(x)
        x = self.upsample(x)

        x = self.resnet(x)
        x = self.bn(x)
        x = self.activation(x)

        # policy
        p = self.pad(x)
        p = self.policy_head(p) # TODO: figure out masking and move selection
        p = self.flatten(p)
        p = self.softmax(p)

        # value
        v = self.pad(x)
        v = self.downsample(v)
        v = self.bn2(v)
        v = self.activation(v)

        v = self.flatten(v)
        v = self.value_head(v)
        v = self.sigmoid(v)


        return p, v