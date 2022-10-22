#%%
import torch 
from torch import nn 
import torch.nn.functional as F
import torchgan.layers as gnn
from torchinfo import summary as torch_summary

from utils import args, device, ConstrainedConv2d, init_weights



def expander(in_channels, out_channels):
    layer = nn.Sequential(
        ConstrainedConv2d(
            in_channels  = in_channels, 
            out_channels = out_channels, 
            kernel_size  = 3,
            padding =      1,
            padding_mode = "reflect"),
        nn.LeakyReLU(),
        nn.Upsample(
            scale_factor = 2, 
            mode = "bilinear"))
    return(layer)



class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv_size = 64
        
        self.start_size = args.image_size // 4
        
        self.seed_in = nn.Sequential(
            nn.Linear(args.seed_size, self.conv_size * self.start_size * self.start_size),
            nn.LeakyReLU())
        
        self.cnn = nn.ModuleList()
        for i in range(2):
            self.cnn.append(expander(self.conv_size, self.conv_size))
            
        self.image_out = nn.Sequential(
            #gnn.SelfAttention2d(input_dims = self.conv_size),
            ConstrainedConv2d(
                in_channels  = self.conv_size, 
                out_channels = 1 if args.gray else 3, 
                kernel_size  = 1),
            nn.Tanh())
        
        self.seed_in.apply(init_weights)
        self.cnn.apply(init_weights)
        self.image_out.apply(init_weights)
        self.to(device)
        
        print("\n\n")
        print(self)
        print()
        print(torch_summary(self, ((2, args.seed_size))))
        print("\n\n")
                
    def go(self, batch_size):
        with torch.no_grad():
            seed = torch.normal(
                mean = torch.zeros([batch_size, args.seed_size]),
                std  = torch.ones( [batch_size, args.seed_size]))
        return(self.forward(seed))
        
    def forward(self, seed):
        seed = seed.to(device)
        x = self.seed_in(seed).reshape(seed.shape[0], self.conv_size, self.start_size, self.start_size)
        x = F.dropout(x, .3)
        #x += torch.normal(
        #    mean = torch.zeros(x.shape),
        #    std  = torch.ones( x.shape)*.25).to(device)
        for l in self.cnn:
            x = l(x)
            x = F.dropout(x, .3)
            #x += torch.normal(
            #    mean = torch.zeros(x.shape),
            #    std  = torch.ones( x.shape)*.25).to(device)
        image = self.image_out(x)
        image = image.cpu()
        return((image.permute(0,2,3,1)+1)/2)
        


if __name__ == "__main__":
    
    gen = Generator()
                

# %%
