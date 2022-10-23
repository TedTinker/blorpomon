#%%

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("--mnist",      type=bool,  default = False) 
parser.add_argument("--gray",       type=bool,  default = False) 
parser.add_argument("--image_size", type=int,   default = 16) 

parser.add_argument("--gen_lr",     type=float, default = .0001) 
parser.add_argument("--dis_lr",     type=float, default = .0001) 
parser.add_argument("--dises",      type=int,   default = 1) 
parser.add_argument("--epochs",     type=int,   default = 20000) 
parser.add_argument("--batch_size", type=int,   default = 64) 
parser.add_argument("--testing",    type=int,   default = 25) 
parser.add_argument("--plotting",   type=int,   default = 25) 
parser.add_argument("--show_plots", type=int,   default = 10) 
parser.add_argument("--keep_gen",   type=int,   default = 1000) 

parser.add_argument("--seed_size",  type=int,   default = 128)
parser.add_argument("--gen_conv",   type=int,   default = 64)  
parser.add_argument("--gen_drop",   type=float, default = .3) 
parser.add_argument("--gen_noise",  type=float, default = .1) 

parser.add_argument("--stat_size",  type=int,   default = 128)
parser.add_argument("--dis_conv",   type=int,   default = 64)  
parser.add_argument("--dis_drop",   type=float, default = .3) 
parser.add_argument("--dis_noise",  type=float, default = .25) 

try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

cmap = "gray" if args.gray else None



import torch 
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n\nDevice: {}.\n\n".format(device))

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    
def init_weights(m):
    try:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass



import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

from random import sample
import cv2
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from keras.datasets import mnist 



def get_buffer():
    if(args.mnist): 
        (train_x, _), (test_x, _) = mnist.load_data()
        buffer = np.concatenate([train_x, test_x], axis = 0)    
        buffer = torch.from_numpy(buffer).reshape((buffer.shape[0], 28, 28, 1))
        buffer = buffer/255
        if(args.gray):  pass 
        else:           buffer = torch.tile(buffer, (1, 1, 1, 3))
    else:
        buffer = []
        files = os.listdir("real_images") ; files.sort()
        for file in files:
            if file not in ["bad_pics", "original.png", "break_picture.py"]:
                pic = image.imread("real_images/" + file)
                pic = pic[:,:,:-1]
                pic = np.clip(pic, 0, 1)
                buffer.append(pic) ; buffer.append(np.flip(pic, axis = 1))
        buffer = torch.from_numpy(np.stack(buffer, axis=0))
        if(args.gray):  buffer = torch.mean(buffer, -1).unsqueeze(-1)
        else:           pass
    buffer = F.interpolate(buffer.permute(0, 3, 1, 2), size = args.image_size).permute(0, 2, 3, 1)
    return(buffer)
    
def get_batch(buffer, batch_size):
    indexes = [i for i in range(len(buffer))]
    indexes = sample(indexes, batch_size)
    batch = torch.clone(buffer[indexes])
    return(batch)

def imshow_shape(image):
    if(image.shape[-1] == 3): pass 
    else: image = torch.tile(image, (1, 1, 1, 3))
    if(len(image.shape) == 3): pass 
    else: image = image.squeeze(0)
    return(image)


            
if __name__ == "__main__":
    buffer = get_buffer()
    plt.imshow(imshow_shape(buffer[0]), cmap = cmap) ; plt.show() ; plt.close() 
    plt.imshow(imshow_shape(buffer[1]), cmap = cmap) ; plt.show() ; plt.close() 
    


def divide_levels(here, plot = plt):
    for h in here:
        plot.axvline(x=h, color = (0,0,0,.2))
        
def plot_losses(loss_acc, show):
    
    xs = [x for x in range(len(loss_acc["gen_train_loss"]))]
    test_xs = loss_acc["test_xs"]
    
    plt.title("Generator Losses")
    plt.plot(xs, loss_acc["gen_train_loss"],      color = "blue", label = "Training loss")
    plt.plot(test_xs, loss_acc["gen_test_loss"],  color = "red",  label = "Testing loss")
    divide_levels(loss_acc["change_level"])
    plt.legend()
    
    plt.savefig("output/gen_loss.png") 
    #if(show): plt.show()
    plt.close()
    
    
    
    for d in range(args.dises):
        plt.title("Discriminator {} Losses".format(d+1))
        plt.plot(xs, loss_acc["dis_train_loss"][d],      color = "blue", label = "Training loss")
        plt.plot(test_xs, loss_acc["dis_test_loss"][d],  color = "red",  label = "Testing loss")
        divide_levels(loss_acc["change_level"])
        plt.legend()
        
        plt.savefig("output/dis_{}_loss.png".format(d+1)) 
        #if(show): plt.show()
        plt.close()
        
        
        
        plt.title("Discriminator {} Accuracy".format(d+1))
        plt.ylim((0, 100))
        plt.plot(xs, loss_acc["dis_real_train_acc"][d],      color = "blue", alpha = .5, label = "Training acc (Real)")
        plt.plot(test_xs, loss_acc["dis_real_test_acc"][d],  color = "red",  alpha = .5,label = "Testing acc (Real)")
        plt.plot(xs, loss_acc["dis_fake_train_acc"][d],      color = "cyan", alpha = .5,label = "Training acc (Fake)")
        plt.plot(test_xs, loss_acc["dis_fake_test_acc"][d],  color = "pink", alpha = .5,label = "Testing acc (Fake)")
        divide_levels(loss_acc["change_level"])
        plt.legend()
        
        plt.savefig("output/dis_{}_acc.png".format(d+1))
        if(show): plt.show()
        plt.close()
    
    

from itertools import chain
from math import floor
        
def plot_examples(reals, fakes, e, show):
    
    fig, axs = plt.subplots(3, 6)
    fig.suptitle("{} Epochs".format(e))
        
    for i, ax in enumerate(list(chain.from_iterable(axs))):
        if(i == 1): ax.title.set_text("Generated")
        if(i == 4): ax.title.set_text("Legitimate")
        num = (i/3 % 2)*3
        num -= 0 if floor(i/3 % 2) == 0 else 3
        num = round(num)
        num += 3*floor(i/6)
        
        buf = fakes if floor(i/3 % 2) == 0 else reals
        image = buf[num]
        ax.imshow(imshow_shape(image), cmap = cmap) 
        ax.axis("off")
    
    fig.tight_layout()
    plt.savefig("output/blorpomon_{}".format(str(e).zfill(10)), dpi=300)
    if(show): plt.show()
    plt.close()
    
    
        
def make_training_vid(fps = 5):
    files = os.listdir("output")
    files.sort()
    files = [f for f in files if f.split("_")[0] == "blorpomon"]
    
    frame = cv2.imread("output/" + files[0]); height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    video = cv2.VideoWriter("output/vid_epochs.avi", fourcc, fps, (width, height))
    for file in files:
        video.write(cv2.imread("output/" + file))
    cv2.destroyAllWindows()
    video.release()
    
    for file in files:
        os.remove("output/" + file)
        
        

import itertools
def make_seeding_vid(gens, seeds, betweens = 5, fps = 5):
    seeds = seeds.unsqueeze(0).permute(0, 2, 1)
    seeds = F.interpolate(seeds, scale_factor = betweens, mode = "linear")
    seeds = seeds.squeeze(0).permute(1, 0)
    images = [gen(seeds).detach() for e, gen in gens]
    original_len = len(images)
    while(len(images) % 10 != 0):
        images.append(torch.ones((images[0].shape)))
        gens.append(("", None))
    
    for i in range(len(images[0])):
        fig, axs = plt.subplots(len(images)//10, 10)
        if(original_len > 10): ax_list = list(itertools.chain.from_iterable(axs))
        else:                   ax_list = axs
        for j, (im, ax) in enumerate(zip(images, ax_list)):
            ax.title.set_text("{} {}".format("" if gens[j][1] == None else "Gen", gens[j][0]))
            ax.title.set_size(5)
            ax.imshow(imshow_shape(im[i]), cmap = cmap)
        for ax in ax_list: ax.axis("off")
        plt.tight_layout()
        plt.savefig("output/seed_{}".format(str(i).zfill(10)), dpi=300)
        plt.close() 
        
    files = os.listdir("output")
    files.sort()
    files = [f for f in files if f.split("_")[0] == "seed"]
    
    frame = cv2.imread("output/" + files[0]); height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    video = cv2.VideoWriter("output/vid_seeds.avi", fourcc, fps, (width, height))
    for file in files:
        video.write(cv2.imread("output/" + file))
    cv2.destroyAllWindows()
    video.release()
    
    for file in files:
        os.remove("output/" + file)
# %%
