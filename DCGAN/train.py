import os
import sys
from solver import solver

train_data_dir = '/data/user/data1/lockeliu/learn/data/GAN/faces'
G_model_path = 'weight/G_cartong.pt'
D_model_path = 'weight/D_cartong.pt'
batch_size = 64
solver.Trainer(train_data_dir, G_model_path, D_model_path, batch_size ).run()
