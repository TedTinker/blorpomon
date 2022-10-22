#%%
import torch 
from torch import nn 
import kornia.color as k
import torchgan.layers as gnn
from torchinfo import summary as torch_summary

from utils import args, device, ConstrainedConv2d, init_weights


    
def contracter(in_channels, out_channels):
    layer = nn.Sequential(
        ConstrainedConv2d(
            in_channels  = in_channels, 
            out_channels = out_channels, 
            kernel_size  = 3,
            padding =      1,
            padding_mode = "reflect"),
        nn.LeakyReLU(),
        nn.MaxPool2d(
            kernel_size = (3,3), 
            stride = (2,2),
            padding = (1,1)))
    return(layer)



class Discriminator(nn.Module):
    
    def __init__(self, var_size = 64):
        super(Discriminator, self).__init__()
        
        self.conv_size = 64
        self.start_size = args.image_size // 4
                
        self.image_in = nn.Sequential(
            nn.BatchNorm2d(1 if args.gray else 3),
            ConstrainedConv2d(
                in_channels  = 1 if args.gray else 3, 
                out_channels = self.conv_size, 
                kernel_size  = 1),
            nn.LeakyReLU(), 
            nn.Dropout(.3))
            #gnn.SelfAttention2d(input_dims = self.conv_size))
        
        self.var_in = nn.Sequential(
            nn.Linear(1, var_size),
            nn.LeakyReLU(),
            nn.Dropout(.3))
        
        self.cnn = nn.ModuleList()
        for i in range(2):
            self.cnn.append(contracter(self.conv_size, self.conv_size))
        
        self.pred_out = nn.Sequential(
            nn.Linear(self.conv_size * self.start_size * self.start_size + var_size, self.conv_size),
            nn.LeakyReLU(),
            nn.Dropout(.3),
            nn.Linear(self.conv_size, 1),
            nn.Tanh())
        
        self.image_in.apply(init_weights)
        self.var_in.apply(init_weights)
        self.cnn.apply(init_weights)
        self.pred_out.apply(init_weights)
        self.to(device)
        
        print("\n\n")
        print(self)
        print()
        print(torch_summary(self, ((2, self.start_size*4, self.start_size*4, 1 if args.gray else 3))))
        print("\n\n")
            
    def forward(self, image, real = None, verbose = False):
        image = image.to(device)
        image = image.permute(0,3,1,2)*2-1
        image += torch.normal(
            mean = torch.zeros(image.shape),
            std  = torch.ones( image.shape)*.25).to(device)
        var = torch.mean(torch.var(image, dim = 0)).unsqueeze(0).unsqueeze(0)
        var = torch.tile(var, (image.shape[0], 1))
        var = self.var_in(var)
        x = self.image_in(image)
        for l in self.cnn:
            x = l(x)
        x = torch.cat([x.flatten(1), var], dim = 1)
        pred = self.pred_out(x).cpu()
        return((pred+1)/2)
        
        

if __name__ == "__main__":
    
    dis = Discriminator()
    

# %%
