import torch.nn as nn


class BaseBlock(nn.Module):
    def __init__(self, c_in, c_out, spatial_downsample):
        super(BaseBlock, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.spatial_downsample = spatial_downsample

        if self.spatial_downsample:
            self.base_block = nn.Sequential(
                self.conv_block(
                    c_in=self.c_in, c_out=self.c_out, kernel_size=3, stride=2, padding=1
                ),
                nn.ReLU(),
                self.conv_block(
                    c_in=self.c_out,
                    c_out=self.c_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        else:
            self.base_block = nn.Sequential(
                self.conv_block(
                    c_in=self.c_in, c_out=self.c_out, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                self.conv_block(
                    c_in=self.c_out,
                    c_out=self.c_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )

        if self.c_in != self.c_out:
            self.pointwise_conv = nn.Conv2d(
                in_channels=self.c_in,
                out_channels=self.c_out,
                kernel_size=1,
                stride=2,
                padding=0,
            )

    def forward(self, x):
        identity_layer = x
        x = self.base_block(x)

        if self.c_in == self.c_out:
            x += identity_layer
        else:
            identity_layer = self.pointwise_conv(identity_layer)
            x += identity_layer

        return x

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, bias=False, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
        )

        return seq_block


class BaseLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(BaseLayer, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.base_layer = nn.Sequential(
            BaseBlock(c_in=self.c_in, c_out=self.c_out, spatial_downsample=True,),
            BaseBlock(c_in=self.c_out, c_out=self.c_out, spatial_downsample=False,),
        )

    def forward(self, x):
        x = self.base_layer(x)

        return x


class ResNet(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(ResNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_classes = num_classes

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )
        self.layer_1 = BaseLayer(c_in=16, c_out=64)
        self.layer_2 = BaseLayer(c_in=64, c_out=128)

        self.gap = nn.AvgPool2d(kernel_size=8)
        self.final_conv = nn.Conv2d(
            in_channels=128, out_channels=10, kernel_size=1, stride=1
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.gap(x)
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    print("Model Summary: \n")
    model = ResNet(num_input_channels=3, num_classes=10)
    summary(model, input_size=(2, 3, 32, 32))
