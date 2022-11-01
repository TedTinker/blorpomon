#%%
from math import floor, ceil

import torch 
from torch import nn 
import torch.nn.functional as F
import torchgan.layers as gnn
from torchinfo import summary as torch_summary

from utils import args, device, ConstrainedConv2d, init_weights 



def expander(in_channels, out_channels):
    layer = nn.Sequential(
        gnn.SelfAttention2d(input_dims = in_channels),
        #ConstrainedConv2d(
        #    in_channels  = in_channels, 
        #    out_channels = out_channels, 
        #    kernel_size  = 3,
        #    padding      = 1,
        #    padding_mode = "reflect"),
        gnn.ResidualBlock2d(
            filters = [in_channels, 2*args.gen_conv, 4*args.gen_conv, out_channels], 
            kernels = [3, 3, 3],
            paddings = [1, 1, 1]),
        nn.LeakyReLU(),
        nn.Upsample(
            scale_factor = 2, 
            mode = "bilinear"))
    return(layer)



class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.verbose = False
        self.epochs = 0
        self.level = 0
        self.color_channels = 1 if args.gray else 3
        
        self.seed_in = nn.Sequential(
            nn.Linear(args.seed_size, args.gen_conv * 4 * 4),
            nn.LeakyReLU())
        
        self.cnn_1 = expander(args.gen_conv, args.gen_conv)
        self.cnn_2 = expander(args.gen_conv, args.gen_conv)
        self.cnn_3 = expander(args.gen_conv, args.gen_conv)
        self.cnn_4 = expander(args.gen_conv, args.gen_conv)
            
        self.image_out = nn.Sequential(
            gnn.SelfAttention2d(input_dims = args.gen_conv),
            ConstrainedConv2d(
                in_channels  = args.gen_conv, 
                out_channels = self.color_channels, 
                kernel_size  = 1),
            nn.Tanh())
        
        self.seed_in.apply(init_weights)
        self.cnn_1.apply(init_weights)
        self.cnn_2.apply(init_weights)
        self.cnn_3.apply(init_weights)
        self.cnn_4.apply(init_weights)
        self.image_out.apply(init_weights)
        self.to(device)
        
    def summary(self):
        image_size = 2**ceil(self.level+2)
        print("\n\nLevel {}, image_size {}".format(self.level, image_size))
        print("\n\n")
        print(self)
        print()
        print(torch_summary(self, ((2, args.seed_size))))
        print("\n\n")
        
    def change_level(self, level):
        self.level = level 
        if(args.freeze):
            freeze_list = [] ; frozen_list = []
            if(level > 1): freeze_list.append(self.cnn[0]) ; frozen_list.append(0)
            if(level > 2): freeze_list.append(self.cnn[1]) ; frozen_list.append(1)
            if(level > 3): freeze_list.append(self.cnn[2]) ; frozen_list.append(2)
            if(level > 4): freeze_list.append(self.cnn[3]) ; frozen_list.append(3)
            for layer in freeze_list:
                layer.requires_grad_(False)
            if(self.verbose): print("Frozen:", frozen_list)
                
    def go(self, batch_size):
        with torch.no_grad():
            seed = torch.normal(
                mean = torch.zeros([batch_size, args.seed_size]),
                std  = torch.ones( [batch_size, args.seed_size]))
        return(self.forward(seed))
        
    def forward(self, seed):
        cnn_list = [self.cnn_1, self.cnn_2, self.cnn_3, self.cnn_4]
        seed = seed.to(device)
        x = self.seed_in(seed).reshape(seed.shape[0], args.gen_conv, 4, 4)
        x = F.dropout(x, args.gen_drop)
        #x += torch.normal(
        #    mean = torch.zeros(x.shape),
        #    std  = torch.ones( x.shape)*args.gen_noise).to(device)
        
        if(self.level == 0): 
            image = self.image_out(x)
        elif(type(self.level) == int):
            for l in range(self.level):
                if(self.verbose): print("\nOrdinary {}\n".format(l))
                x = cnn_list[l](x)
                x = F.dropout(x, args.gen_drop)
            image = self.image_out(x)
        else:
            level = floor(self.level)
            alpha = self.level - level
            for l in range(level):
                if(self.verbose): print("\nOrdinary {}\n".format(l))
                x = cnn_list[l](x)
                x = F.dropout(x, args.gen_drop)
            old_image = F.interpolate(self.image_out(x), scale_factor = 2, mode = "nearest")
            if(self.verbose): print("\nProgressive {}\n".format(level))
            x = cnn_list[level](x)
            x = F.dropout(x, args.gen_drop)
            new_image = self.image_out(x)
            image = (1 - alpha) * old_image + alpha * new_image

        image = image.cpu()
        torch.cuda.empty_cache()
        return((image.permute(0,2,3,1)+1)/2)
        


if __name__ == "__main__":
    
    gen = Generator() ; gen.verbose = True
                
    for level in [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
        print("\n\nLEVEL {}\n\n".format(level))
        gen.change_level(level)
        gen.summary()
# %%
