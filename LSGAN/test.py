import os
import sys
from solver import solver

train_data_dir = '/data/user/data1/lockeliu/learn/data/GAN/mnist'
G_model_path = 'weight/Gv2.pt'
D_model_path = 'weight/Dv2.pt'
test = solver.Trainer(train_data_dir, G_model_path, D_model_path, 100 )
test.build_model()
test.show_result(100000)
