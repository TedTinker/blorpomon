import torch 
from torch import nn 
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



conv_size = 64

class Discriminator(nn.Module):
    
    def __init__(self, norm_size = 64, mnist = False):
        super(Discriminator, self).__init__()
        
        self.start_size = args.image_size // 4
                
        self.image_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels  = 1 if args.gray else 3, 
                out_channels = conv_size, 
                kernel_size  = 1),
            nn.LeakyReLU(), 
            nn.Dropout(.3))
        
        self.norm_in = nn.Sequential(
            nn.Linear(1, norm_size),
            nn.LeakyReLU())
        
        self.cnn = nn.Sequential(
            contracter(conv_size, conv_size),
            contracter(conv_size, conv_size))
        
        self.pred_out = nn.Sequential(
            nn.Linear(conv_size * self.start_size * self.start_size + norm_size, conv_size),
            nn.LeakyReLU(),
            nn.Dropout(.3),
            nn.Linear(conv_size, 1),
            nn.Tanh())
        
        self.image_in.apply(init_weights)
        self.norm_in.apply(init_weights)
        self.cnn.apply(init_weights)
        self.pred_out.apply(init_weights)
        self.to(device)
        
        print("\n\n")
        print(self)
        print()
        print(torch_summary(self, ((2, self.start_size*4, self.start_size*4, 1 if args.gray else 3))))
        print("\n\n")
            
    def forward(self, image):
        image = image.to(device)
        image = (image.permute(0,3,1,2)*2)-1
        x = self.image_in(image)
        x = self.cnn(x)
        norm = torch.linalg.norm(image.detach(), dim=(1,2,3))
        norm = self.norm_in(norm.unsqueeze(1))
        x = torch.cat([x.flatten(1), norm], dim = 1)
        pred = self.pred_out(x).cpu()
        return((pred+1)/2)
        
        

if __name__ == "__main__":
    
    dis = Discriminator()
    
