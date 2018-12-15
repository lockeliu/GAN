import torch
import os
from torchvision import transforms
from comm import load_data
from comm.misc import progress_bar
from model import generator,discriminator
from torch import nn
from PIL import Image

class Trainer(object):
    def __init__(self, train_data_dir, G_model_path, D_model_path, train_batch_size = 64, epoch = 100, lr = 0.0002, gpu_num=4):
        self.train_batch_size = train_batch_size
        self.train_data_dir = train_data_dir
        self.epoch = epoch
        self.lr = lr
        self.gpu_list = [ gpu_id for gpu_id in range( gpu_num ) ]
        self.G_model_path = G_model_path
        self.D_model_path = D_model_path
        self.img_size = 64 

    def build_model(self):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
           ])

        trainset = load_data.MyDataset( self.train_data_dir, transform );
        self.training_loader = torch.utils.data.DataLoader( trainset , batch_size = self.train_batch_size , shuffle = True, num_workers=20 )

        # G
        self.G_net = generator.Generator( input_dim = 100, output_size = self.img_size, output_dim = 3, model_path = self.G_model_path )
        self.G_net = self.G_net.cuda()
        self.G_model = torch.nn.DataParallel( self.G_net, self.gpu_list )
        self.G_optimizer = torch.optim.Adam( self.G_model.parameters(), lr=self.lr )
        #self.G_scheduler = torch.optim.lr_scheduler.StepLR(self.G_optimizer, step_size=20, gamma=0.5)
        
        # D
        self.D_net = discriminator.Discriminator( input_dim = 3, input_size = self.img_size, output_dim = 1, model_path = self.D_model_path )
        self.D_net = self.D_net.cuda()
        self.D_model = torch.nn.DataParallel( self.D_net, self.gpu_list )
        self.D_optimizer = torch.optim.Adam( self.D_model.parameters(), lr=self.lr )
        #self.D_scheduler = torch.optim.lr_scheduler.StepLR(self.D_optimizer, step_size=20, gamma=0.5)

        self.MSE_loss = nn.MSELoss();

        self.test_noise = torch.randn( ( 100, 100) )

        #print (self.G_net)
        #print (self.D_net)

    def save(self):
        self.G_net.savemodel()
        self.D_net.savemodel()

    def show_result(self, epoch):
        dir_path = 'show_result'
        if os.path.exists( dir_path ) == False:
            os.makedirs( dir_path)

        self.G_model.eval()
        target = self.G_model( self.test_noise ).cpu()
        
        img_size = self.img_size + 10
        img_result = Image.new( 'RGB', ( img_size * 10, img_size * 10 ), (255,255,255) ) 
        for i in range( 10 ):
            for j in range( 10 ):
                img = target[i*10 +j];
                img = img / 2 + 0.5
                img = transforms.ToPILImage()(img)
                img_result.paste( img, ( i * img_size + 5, j * img_size + 5 ) )
        img_result.save( os.path.join( dir_path, str(epoch) + '.jpg') )

    def train(self):
        self.G_model.train()
        self.D_model.train()
        D_train_loss = 0
        G_train_loss = 0
        for batch_num, data in enumerate(self.training_loader):
            batch_size = data.size()[0]
            

            y_real_ = torch.ones( (batch_size, 1) )
            y_fake_ = torch.zeros( (batch_size,1 ) )
            y_real_, y_fake_, data = y_real_.cuda(), y_fake_.cuda(), data.cuda()
            # train D
            self.D_model.zero_grad()

            D_real_loss = self.MSE_loss( self.D_model( data ), y_real_ )

            noise = torch.randn( ( batch_size, 100 ) )
            D_fake_loss = self.MSE_loss( self.D_model( self.G_model( noise) ), y_fake_ )

            D_loss = D_real_loss + D_fake_loss 
            D_train_loss += D_loss.item()
            D_loss.backward();
            self.D_optimizer.step()
    
            # train G
            self.G_model.zero_grad()

            noise = torch.randn( ( batch_size, 100 ) )
            G_loss = self.MSE_loss( self.D_model( self.G_model( noise ) ), y_real_ );
            G_train_loss += G_loss.item()
            G_loss.backward();
            self.G_optimizer.step()

            progress_bar(batch_num, len(self.training_loader), 'D_Loss: %.4f G_Loss: %.4f' % (  D_train_loss / (batch_num + 1), G_train_loss / (batch_num + 1) ))

        print("    Average D_Loss: {:%.4f} G_Loss: {:%.4f} " % ( D_train_loss / len(self.training_loader), G_train_loss / len(self.training_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.epoch + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            #self.G_scheduler.step(epoch)
            #self.D_scheduler.step(epoch)
            self.show_result(epoch)
            self.save()


#train_data_dir = '/data/user/data1/lockeliu/learn/data/GAN/mnist'
#G_model_path = 'weight/G.pt'
#D_model_path = 'weight/D.pt'
#Trainer(train_data_dir, G_model_path, D_model_path, 256 ).run() 
