#%%
from math import floor, ceil

import torch 
from torch import nn 
import torch.nn.functional as F
import kornia.color as k
import torchgan.layers as gnn
from torchinfo import summary as torch_summary

from utils import args, device, ConstrainedConv2d, init_weights


    
def contracter(in_channels, out_channels):
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
        nn.AvgPool2d(
            kernel_size = 2),
        nn.LeakyReLU())
    return(layer)



def get_stats(image):
    # Statistics for each color of each image
    c_mean   = image.mean(2).mean(2)
    c_q      = torch.quantile(image.flatten(2), q = torch.tensor([.01, .05, .15, .25, .35, .5, .65, .75, .85, .95, .99]).to(device), dim = 2).permute(1, 2, 0).flatten(1)
    c_mode   = image.mode(2)[0].mode(2)[0]
    c_var    = torch.var(image, dim = (2, 3)) 
    
    # Statistics for entire batch
    b_mean   = torch.tile(c_mean.mean(0).unsqueeze(0), (image.shape[0], 1))
    b_q      = torch.tile(torch.quantile(image.permute(1, 0, 2, 3).flatten(1), q = torch.tensor([.01, .05, .15, .25, .35, .5, .65, .75, .85, .95, .99]).to(device), dim = 1).flatten(0).unsqueeze(0), (image.shape[0], 1))
    b_mode   = torch.tile(c_mode.mode(0)[0].unsqueeze(0), (image.shape[0], 1))
    b_var    = torch.tile(torch.var(image, dim = (0,1,2,3)).unsqueeze(0).unsqueeze(0), (image.shape[0], 1))
    
    #for stat in [c_mean, c_q, c_mode, c_var, b_mean, b_q, b_mode, b_var]:
    #    print(stat.shape)
    
    stats = torch.cat([c_mean, c_q, c_mode, c_var, 
                       b_mean, b_q, b_mode, b_var], dim = 1)
    return(stats)



stat_quantity = get_stats(
    torch.zeros((1, 1 if args.gray else 3, 4, 4)).to(device)).shape[1]



class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.verbose = False
        self.epochs = 0
        self.level = 0
        self.color_channels = 1 if args.gray else 3
        
        self.stats_in = nn.Sequential(
            nn.Linear(
                stat_quantity, 
                args.stat_size),
            nn.LeakyReLU(),
            nn.Dropout(args.dis_drop),
            nn.Linear(
                args.stat_size, 
                args.stat_size),
            nn.LeakyReLU(),
            nn.Dropout(args.dis_drop))
                
        self.image_in = nn.Sequential(
            nn.BatchNorm2d(self.color_channels),
            ConstrainedConv2d(
                in_channels  = self.color_channels, 
                out_channels = args.dis_conv, 
                kernel_size  = 1),
            nn.LeakyReLU(), 
            nn.Dropout(args.dis_drop),
            gnn.SelfAttention2d(input_dims = args.dis_conv))
        
        self.cnn_1 = contracter(args.dis_conv, args.dis_conv)
        self.cnn_2 = contracter(args.dis_conv, args.dis_conv)
        self.cnn_3 = contracter(args.dis_conv, args.dis_conv)
        self.cnn_4 = contracter(args.dis_conv, args.dis_conv)
        
        self.pred_out = nn.Sequential(
            nn.Linear(args.dis_conv * 4 * 4 + args.stat_size, args.dis_conv),
            nn.LeakyReLU(),
            nn.Dropout(args.dis_drop),
            nn.Linear(args.dis_conv, 1),
            nn.Tanh())
        
        self.image_in.apply(init_weights)
        self.stats_in.apply(init_weights)
        self.cnn_1.apply(init_weights)
        self.cnn_2.apply(init_weights)
        self.cnn_3.apply(init_weights)
        self.cnn_4.apply(init_weights)
        self.pred_out.apply(init_weights)
        self.to(device)
        
    def summary(self):
        image_size = 2**ceil(self.level+2)
        print("\n\nLevel {}, image_size {}".format(self.level, image_size))
        print("\n\n")
        print(self)
        print()
        print(torch_summary(self, ((2, image_size, image_size, self.color_channels))))
        print("\n\n")
        
    def change_level(self, level):
        self.level = level 
        if(args.freeze):
            freeze_list = [] ; frozen_list = []
            if(level > 1): freeze_list.append(self.cnn[-1]) ; frozen_list.append(-1)
            if(level > 2): freeze_list.append(self.cnn[-2]) ; frozen_list.append(-2)
            if(level > 3): freeze_list.append(self.cnn[-3]) ; frozen_list.append(-3)
            if(level > 4): freeze_list.append(self.cnn[-4]) ; frozen_list.append(-4)
            for layer in freeze_list:
                layer.requires_grad_(False)
            if(self.verbose): print("Frozen:", frozen_list)
            
    def forward(self, image):
        cnn_list = [self.cnn_1, self.cnn_2, self.cnn_3, self.cnn_4]
        image = image.to(device)
        image = image.permute(0,3,1,2)*2-1
        image += torch.normal(
            mean = torch.zeros(image.shape),
            std  = torch.ones( image.shape)*args.dis_noise).to(device)
        
        stats = get_stats(image)
        stats = self.stats_in(stats)
        x = self.image_in(image)
                
        if(self.level == 0): pass
        else:
            if(type(self.level) == int):
                for l in [-(self.level - i) for i in range(self.level)]:
                    if(self.verbose): print("\nOrdinary {}\n".format(l))
                    x = cnn_list[l](x)
                    x = F.dropout(x, args.gen_drop)
            else:
                level = floor(self.level)
                alpha = self.level - level
                old_image = F.interpolate(image, scale_factor = .5, mode = "nearest")
                old_x = self.image_in(old_image)
                if(self.verbose): print("\nProgressive {}\n".format(-level - 1))
                new_x = cnn_list[-level - 1](x)
                new_x = F.dropout(new_x, args.gen_drop)
                x = (1 - alpha) * old_x + alpha * new_x
                for l in [-(level - i + 1) + 1 for i in range(level)]:
                    if(self.verbose): print("\nOrdinary {}\n".format(l))
                    x = cnn_list[l](x)
                    x = F.dropout(x, args.gen_drop)
            
        x = torch.cat([x.flatten(1), stats], dim = 1)
        pred = self.pred_out(x).cpu()
        torch.cuda.empty_cache()
        return((pred+1)/2)
        
        

if __name__ == "__main__":
    
    dis = Discriminator() ; dis.verbose = True
    
    #for level in [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
    for level in [0, .5]:
        print("LEVEL {}".format(level))
        dis.change_level(level)
        dis.summary()
# %%
