import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module ):
    def __init__(self, input_dim, input_size, output_dim, model_path = 'weight/Discriminator_weight.pt' ):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.output_dim = output_dim
        self.inp = self.input_size // 4
        self.model_path = model_path

        self.conv = nn.Sequential(
                nn.Conv2d( self.input_dim, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d( 64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )

        self.fc = nn.Sequential(
                nn.Linear( 128 * self.inp * self.inp, 1024 ),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_dim),
                nn.Sigmoid()
            )

        self.load()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * self.inp * self.inp )
        x = self.fc( x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


    def savemodel(self):
        torch.save( self.state_dict(), self.model_path );

    def load(self):
        if os.path.exists(self.model_path):
            print("load D_weight")
            model = torch.load(self.model_path)
            self.load_state_dict(model)
        else:
            print("init D_weight")
            self._initialize_weights()

