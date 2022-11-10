import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.spaconv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=5, dilation=5)
        self.spabn1 = nn.BatchNorm2d(64)
        self.spaconv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.spabn2 = nn.BatchNorm2d(64)
        self.spaconv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.spabn3 = nn.BatchNorm2d(64)
        self.spaconv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.spabn4 = nn.BatchNorm2d(128)
        self.spaconv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.spabn5 = nn.BatchNorm2d(128)
        self.spaconv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.spabn6 = nn.BatchNorm2d(256)
        self.spaconv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.spabn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(1024)
        self.conv9 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(1024)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(1, stride=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

        self.linear_scorein = nn.Linear(1024,128)
        self.linear_scoreout = nn.Linear(128, 1)

    def forward(self, x):

        spax = self.spaconv1(x)
        spax = self.spabn1(spax)
        spax = self.relu(spax)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        residual = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)

        residual = x
        residual = F.pad(residual, (0, 0, 0, 0, 0, 64))
        residual = self.maxpool(residual)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x += residual
        x = self.relu(x)

        residual = x
        residual = F.pad(residual, (0, 0, 0, 0, 0, 128))
        residual = self.maxpool(residual)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x += residual
        x = self.relu(x)

        residual = spax
        spax = self.spaconv2(spax)
        spax = self.spabn2(spax)
        spax = self.relu(spax)
        spax = self.spaconv3(spax)
        spax = self.spabn3(spax)
        spax += residual
        spax = self.relu(spax)

        residual = spax
        residual = F.pad(residual, (0, 0, 0, 0, 0, 64))
        residual = self.maxpool(residual)
        spax = self.spaconv4(spax)
        spax = self.spabn4(spax)
        spax = self.relu(spax)
        spax = self.spaconv5(spax)
        spax = self.spabn5(spax)
        spax += residual
        spax = self.relu(spax)

        residual = spax
        residual = F.pad(residual, (0, 0, 0, 0, 0, 128))
        residual = self.maxpool(residual)
        spax = self.spaconv6(spax)
        spax = self.spabn6(spax)
        spax = self.relu(spax)
        spax = self.spaconv7(spax)
        spax = self.spabn7(spax)
        spax += residual
        spax = self.relu(spax)

        out = torch.cat([x,spax], dim = 1)

        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.relu(out)

        out = self.avg_pooling(out)
        out = self.flat(out)

        score_out = self.linear_scorein(out)
        score_out = self.linear_scoreout(score_out)

        return score_out

