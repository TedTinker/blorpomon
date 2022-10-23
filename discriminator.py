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
    torch.zeros((1, 1 if args.gray else 3, args.image_size, args.image_size)).to(device)).shape[1]



class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.start_size = args.image_size // 4
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
        
        self.cnn = nn.ModuleList()
        for i in range(2):
            self.cnn.append(contracter(args.dis_conv, args.dis_conv))
        
        self.pred_out = nn.Sequential(
            nn.Linear(args.dis_conv * self.start_size * self.start_size + args.stat_size, args.dis_conv),
            nn.LeakyReLU(),
            nn.Dropout(args.dis_drop),
            nn.Linear(args.dis_conv, 1),
            nn.Tanh())
        
        self.image_in.apply(init_weights)
        self.stats_in.apply(init_weights)
        self.cnn.apply(init_weights)
        self.pred_out.apply(init_weights)
        self.to(device)
        
        print("\n\n")
        print(self)
        print()
        print(torch_summary(self, ((2, self.start_size*4, self.start_size*4, self.color_channels))))
        print("\n\n")
            
    def forward(self, image, real = None, verbose = False):
        image = image.to(device)
        image = image.permute(0,3,1,2)*2-1
        image += torch.normal(
            mean = torch.zeros(image.shape),
            std  = torch.ones( image.shape)*args.dis_noise).to(device)
        
        stats = get_stats(image)
        stats = self.stats_in(stats)
        
        x = self.image_in(image)
        for l in self.cnn:
            x = l(x)
        x = torch.cat([x.flatten(1), stats], dim = 1)
        pred = self.pred_out(x).cpu()
        return((pred+1)/2)
        
        

if __name__ == "__main__":
    
    dis = Discriminator()
    

# %%
