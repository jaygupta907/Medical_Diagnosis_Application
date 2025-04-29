import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block used in a deep network to maintain learning flow.

    Args:
        in_channels (int): The number of input channels to the block.
        out_channels (int): The number of output channels from the block.

    Returns:
        None
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (Tensor): Input tensor to the residual block.

        Returns:
            Tensor: Output tensor after applying the residual block.
        """
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class UNet(nn.Module):
    """
    U-Net architecture used for segmentation tasks.

    Args:
        in_channels (int): The number of input channels (default: 3).
        out_channels (int): The number of output channels (default: 1).

    Returns:
        None
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        def conv_block(in_channels, out_channels):
            """
            Helper function for creating the convolutional block with residuals.

            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.

            Returns:
                Sequential: A sequential block of residuals.
            """
            return nn.Sequential(
                ResidualBlock(in_channels, out_channels),
                ResidualBlock(out_channels, out_channels)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 1024),
            ResidualBlock(1024, 1024)
        )

        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, 1)

        # Max pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """
        Forward pass through the U-Net model.

        Args:
            x (Tensor): Input tensor to the U-Net model.

        Returns:
            Tensor: Output segmentation map from the U-Net.
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Final output with sigmoid
        out = self.final_conv(dec1)
        out = torch.sigmoid(out)
        return out

class resblock(nn.Module):
    """
    Residual block with convolutional layers and skip connection.

    Args:
        in_planes (int): Number of input channels to the block.
        out_planes (int): Number of output channels from the block.
        downsample (bool): Whether to apply downsampling in the block.

    Returns:
        None
    """
    def __init__(self, in_planes, out_planes, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
            self.identity = nn.Sequential()

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_planes)
        self.batch_norm2 = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        """
        Forward pass through the residual block.

        Args:
            input (Tensor): Input tensor to the block.

        Returns:
            Tensor: Output tensor after applying the residual block.
        """
        identity = self.identity(input)
        input = self.conv1(input)
        input = self.batch_norm1(input)
        input = nn.ReLU()(input)
        input = self.conv2(input)
        input = self.batch_norm2(input)
        input = input + identity
        output = nn.ReLU()(input)
        return output

class resnet(nn.Module):
    """
    ResNet model for classification tasks with residual blocks.

    Args:
        in_planes (int): Number of input channels to the network.
        outputs (int): Number of output classes.

    Returns:
        None
    """
    def __init__(self, in_planes, outputs):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )
        self.layer5 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(512, outputs)

    def forward(self, input):
        """
        Forward pass through the ResNet model.

        Args:
            input (Tensor): Input tensor to the ResNet model.

        Returns:
            Tensor: Output predictions from the model.
        """
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.avg_pooling(output)
        output = output.view(input.size(0), -1)
        output = self.fully_connected(output)
        output = torch.sigmoid(output)
        return output
