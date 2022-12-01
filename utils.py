#%%
# Thinks to do: Residual Blocks!

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("--images",     type=str,   default = "faces") # mnist, pokemon, faces
parser.add_argument("--gray",       type=bool,  default = False) 

parser.add_argument("--gen_lr",     type=dict,  default = {
    (0, int)   : .00003, 
    (0, float) : .00003, (1, int) : .00003, 
    (1, float) : .00003, (2, int) : .00003, 
    (2, float) : .00003, (3, int) : .00003, 
    (3, float) : .00003, (4, int) : .00003}) 
parser.add_argument("--dis_lr",     type=dict,  default = {
    (0, int)   : .00001, 
    (0, float) : .00001, (1, int) : .00001, 
    (1, float) : .00001, (2, int) : .00001, 
    (2, float) : .00001, (3, int) : .00001, 
    (3, float) : .00001, (4, int) : .00001}) 
parser.add_argument("--dises",      type=int,   default = 5) 
parser.add_argument("--freeze",     type=tuple, default = (True, True))
parser.add_argument("--epochs",     type=dict,  default = {
    (0, int)   : 20000, 
    (0, float) : 20000, (1, int) : 20000, 
    (1, float) : 20000, (2, int) : 20000, 
    (2, float) : 20000, (3, int) : 20000, 
    (3, float) : 0, (4, int) : 0}) 
parser.add_argument("--batch_sizes",type=dict,  default = {    
    (0, int)   : (64, 64), 
    (0, float) : (64, 64), (1, int) : (64, 64), 
    (1, float) : (64, 64), (2, int) : (64, 64), 
    (2, float) : (64, 64), (3, int) : (64, 64), 
    (3, float) : (64, 64), (4, int) : (64, 64)}) 
parser.add_argument("--prev_batch", type=tuple, default = (2,)*8)
parser.add_argument("--mismatch",   type=int,   default = 1)

parser.add_argument("--plotting",   type=int,   default = 500) 
parser.add_argument("--show_plots", type=int,   default = 10) 
parser.add_argument("--keep_gen",   type=int,   default = 5000) 

parser.add_argument("--seed_size",  type=int,   default = 256)
parser.add_argument("--gen_conv",   type=int,   default = 128)  
parser.add_argument("--gen_drop",   type=float, default = .2) 
parser.add_argument("--gen_noise",  type=float, default = .1) 

parser.add_argument("--stat_size",  type=int,   default = 256)
parser.add_argument("--dis_conv",   type=int,   default = 128)  
parser.add_argument("--dis_drop",   type=float, default = .2) 
parser.add_argument("--dis_noise",  type=float, default = .1) 

try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

cmap = "gray" if args.gray else None



import torch 
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(.99, device=0)

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

from itertools import chain, product
from math import floor
from random import sample
import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import image
import matplotlib.pyplot as plt
mpl.rcParams['agg.path.chunksize'] = 9999999999999999



def get_buffer(size, images = args.images, gray = args.gray):
    if(images == "mnist"): 
        buffer = torch.load("real_images/mnist.pt")
        if(gray):  pass 
        else:      buffer = torch.tile(buffer, (1, 1, 1, 3))
        
    if(images == "faces"): 
        buffer = torch.load("real_images/faces.pt")
        if(gray):  buffer = torch.mean(buffer, -1).unsqueeze(-1)
        else:      pass
        
    if(images == "pokemon"):
        buffer = []
        files = os.listdir("real_images/pokemon") ; files.sort()
        for file in files:
            if file not in ["bad_pics", "original.png", "break_picture.py"]:
                pic = image.imread("real_images/pokemon/" + file)
                pic = pic[:,:,:-1]
                pic = np.clip(pic, 0, 1)
                buffer.append(pic) ; buffer.append(np.flip(pic, axis = 1))
        buffer = torch.from_numpy(np.stack(buffer, axis=0))
        if(gray):  buffer = torch.mean(buffer, -1).unsqueeze(-1)
        else:      pass
        
    buffer = F.interpolate(buffer.permute(0, 3, 1, 2), size = size, mode = "bilinear").permute(0, 2, 3, 1)
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
    buffer = get_buffer(64, "mnist")
    plt.imshow(imshow_shape(buffer[0]), cmap = cmap) ; plt.show() ; plt.close() 
    plt.imshow(imshow_shape(buffer[1]), cmap = cmap) ; plt.show() ; plt.close() 
    buffer = get_buffer(64, "faces")
    plt.imshow(imshow_shape(buffer[0]), cmap = cmap) ; plt.show() ; plt.close() 
    plt.imshow(imshow_shape(buffer[1]), cmap = cmap) ; plt.show() ; plt.close() 
    buffer = get_buffer(64, "pokemon")
    plt.imshow(imshow_shape(buffer[0]), cmap = cmap) ; plt.show() ; plt.close() 
    plt.imshow(imshow_shape(buffer[1]), cmap = cmap) ; plt.show() ; plt.close() 

    
    
# Monitor GPU memory.
def get_free_mem(string = ""):
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("\n{}: {}.\n".format(string, f))
    pass 

# Remove from GPU memory.
def delete_these(verbose = False, *args):
    if(verbose): get_free_mem("Before deleting")
    del args
    torch.cuda.empty_cache()
    #torch.cuda.empty_cache()
    if(verbose): get_free_mem("After deleting")    



def divide_levels(here, plot = plt):
    for h in here:
        plot.axvline(x=h, color = (0,0,0,.2))
        
def get_min_median_max(loss_acc, name):
    mins    = [l[0] for l in loss_acc[name]]
    medians = [l[1] for l in loss_acc[name]]
    maxs    = [l[2] for l in loss_acc[name]]
    return(mins, medians, maxs)

line_alpha = .2 ; fill_alpha = .2



def plots(loss_acc, reals, fakes, e, level, size, show):
    
    # Plot examples
    fig = plt.figure(figsize = (15, 10))
    gs = fig.add_gridspec(7, 7, width_ratios = [1,1,1,.1,1,1,1], height_ratios = [1,1,1,.1,1,1,1])
    fig.suptitle("{} Epochs (Level {} : {} Pixels)".format(e, level, size))
    
    ax = fig.add_subplot(gs[0:3, 0:3]) ; ax.set_xticks([]) ; ax.set_yticks([])
    ax = fig.add_subplot(gs[0:3, 4:7]) ; ax.set_xticks([]) ; ax.set_yticks([])
        
    positions = list(product([0,1,2], [0,1,2,4,5,6]))
    for pos in positions:
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        if(pos == (0, 1)): ax.title.set_text("Generated")
        if(pos == (0, 5)): ax.title.set_text("Legitimate")
        if(pos[1] in [0,1,2]): buf = fakes
        else:                  buf = reals
        image = buf[3*pos[0] + pos[1] - (0 if pos[1] in [0,1,2] else 4)]
        ax.imshow(imshow_shape(image), cmap = cmap) 
        ax.axis("off")
    
    # Plot losses
    xs = [x for x in range(1, 1+len(loss_acc["gen_loss"]))]
    loss_mins, loss_medians, loss_maxs = \
        get_min_median_max(loss_acc, "dis_loss")
    real_acc_mins, real_acc_medians, real_acc_maxs = \
        get_min_median_max(loss_acc, "dis_real_acc")
    fake_acc_mins, fake_acc_medians, fake_acc_maxs = \
        get_min_median_max(loss_acc, "dis_fake_acc")
    
    ax = fig.add_subplot(gs[4:7, 0:3])
    ax.title.set_text("Generator and Discriminator Losses")
    ax.set_xlabel("Epochs") ; ax.set_ylabel("Loss")
    ax.plot(xs, loss_acc["gen_loss"], color = "red", label = "Gen", alpha = line_alpha)
    ax.fill_between(xs, loss_mins, loss_maxs, color = "blue", alpha = fill_alpha, linewidth = 0, label = "Dis")
    #ax.plot(xs, loss_medians, color = "blue", alpha = line_alpha, label = "Dis")
    divide_levels(loss_acc["change_level"])
    ax.legend(loc = "center left")
    
    ax = fig.add_subplot(gs[4:7, 4:7])
    ax.set_xlabel("Epochs") ; ax.set_ylabel("Percent Accurate") ; ax.set_yticks(ticks = [i*10 for i in range(11)])
    ax.title.set_text("Discriminator Accuracy")
    ax.set_ylim((0, 100)) 
    ax.axhline(y = 50, color = (0, 0, 0, .2), linestyle = "--")
    ax.fill_between(xs, fake_acc_mins, fake_acc_maxs, color = "red", alpha = fill_alpha, linewidth = 0, label = "Fake")
    #ax.plot(xs, fake_acc_medians, color = "red", alpha = line_alpha, label = "Fake")
    ax.fill_between(xs, real_acc_mins, real_acc_maxs, color = "blue", alpha = fill_alpha, linewidth = 0, label = "Real")
    #ax.plot(xs, real_acc_medians, color = "blue", alpha = line_alpha, label = "Real")
    divide_levels(loss_acc["change_level"])
    ax.legend(loc = "center left")
    
    # Empty space 
    positions = list(product([0,1,2,3,4,5], [3]))
    for pos in positions:
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.axis("off")
    
    plt.savefig("output/blorpomon_{}".format(str(e).zfill(10)), dpi=200)
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
        
        

def make_seeding_vid(betweens = 5, fps = 5):
    seeds = torch.load("output/seeds.pt")
    seed_list = list(torch.split(seeds, 1, dim = 0))
    seed_list.append(seed_list[0])
    seeds = seed_list[0]
    for seed, next_seed in zip(seed_list[:-1], seed_list[1:]):
        between_list = [i/(betweens+1) for i in range(1, betweens+2)]
        for b in between_list:
            seeds = torch.cat([seeds, (1-b)*seed + b*next_seed])
    
    gen_files = os.listdir("output/gens") ; gen_files.sort()
    gens = [torch.load("output/gens/" + file) for file in gen_files]
    images = [gen(seeds).detach() for gen in gens]
    original_len = len(images)
    while(len(images) % 10 != 0):
        images.append(torch.ones((images[0].shape)))
        gens.append(None)
    
    for i in range(len(images[0])):
        fig, axs = plt.subplots(len(images)//10, 10)
        if(original_len > 10): ax_list = list(chain.from_iterable(axs))
        else:                   ax_list = axs
        for j, (im, ax) in enumerate(zip(images, ax_list)):
            gen = gens[j]
            ax.title.set_text("{}".format(
                "" if gen == None else "Gen {}\nLevel {}".format(gen.epochs, round(gen.level, 3))))
            ax.title.set_size(5)
            ax.imshow(imshow_shape(im[i]), cmap = cmap)
        for ax in ax_list: ax.axis("off")
        #plt.tight_layout()
        plt.savefig("output/seed_{}".format(str(i).zfill(10)), dpi=200)
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
