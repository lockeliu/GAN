import os
import sys
import torch
import torch.nn as nn

class Generator( nn.Module ):
    def __init__(self, input_dim, output_size, output_dim, model_path = 'weight/generater_weight.pt'):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        self.output_dim = output_dim
        self.inp = self.output_size // 4;
        self.model_path = model_path

        self.fc = nn.Sequential(
                nn.Linear( self.input_dim, 1024),
                nn.BatchNorm1d( 1024 ),
                nn.ReLU(),
                nn.Linear(1024, 128 * self.inp * self.inp ), 
                nn.BatchNorm1d( 128 * self.inp * self.inp ),
                nn.ReLU(),
            )

        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
                nn.Tanh(),
            )
        
            

        self.load()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, self.inp, self.inp)
        x = self.deconv(x)
        return x;


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
            print("load G_weight")
            model = torch.load(self.model_path)
            self.load_state_dict(model)
        else:
            print("init G_weight")
            self._initialize_weights()
