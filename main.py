#%%
import os
import shutil
import enlighten 
from copy import deepcopy
from math import floor
import numpy as np
from random import sample

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torchgan.losses import WassersteinGeneratorLoss as WG
from torchgan.losses import WassersteinDiscriminatorLoss as WD

from utils import args, get_buffer, get_batch, plots, make_training_vid, make_seeding_vid
from generator import Generator
from discriminator import Discriminator


    
def epoch(buffer, batch_sizes, prev_batches, gen, gen_opt, dises, dis_opts):
    
    wg = WG()
    
    gen_opt.zero_grad()
    fake_batch = gen.go(batch_sizes[0])
    guesses = torch.cat([dis(fake_batch) for dis in dises], dim = 1)
    gen_loss = wg(guesses)
    
    gen_loss.backward()
    gen_opt.step()
        
    wd = WD()
    
    real_batch = get_batch(buffer, batch_sizes[1])
    with torch.no_grad(): 
        fake_batch = gen.go(batch_sizes[1])
    
    prev_batch_keys = list(prev_batches.keys())
    for i in prev_batch_keys:
        if(prev_batches[i] != []):
            fake_batch = torch.cat([prev_batches[i][i], fake_batch], dim = 0)
    prev_batch_keys.reverse()
    for i in prev_batch_keys:
        if(i+1 in prev_batch_keys):
            prev_batches[i+1] = prev_batches[i]
    prev_batches[0] = []
    for size in args.prev_batch:
        prev_batches[0].append(fake_batch[-size:])
        fake_batch = fake_batch[:-size]
    real_batch = real_batch[:fake_batch.shape[0]]        
    
    if(args.mismatch < real_batch.shape[0]):
        random_to_swap = sample([i for i in range(len(real_batch))], args.mismatch)
        mismatch_real = real_batch[random_to_swap]
        mismatch_fake = fake_batch[random_to_swap]
        real_batch[random_to_swap] = mismatch_fake 
        fake_batch[random_to_swap] = mismatch_real
        
    losses = [] ; real_accs = [] ; fake_accs = []
    for dis, dis_opt in zip(dises, dis_opts):
    
        dis_opt.zero_grad()
        real_guesses = dis(real_batch.detach())
        real_guesses_l = [round(g[0]) for g in real_guesses.tolist()]
        real_acc = sum([1 if g == 1 else 0 for g in real_guesses_l]) / len(real_guesses)
        
        dis_opt.zero_grad()
        fake_guesses = dis(fake_batch.detach())
        fake_guesses_l = [round(g[0]) for g in fake_guesses.tolist()]
        fake_acc = sum([1 if g == 0 else 0 for g in fake_guesses_l]) / len(fake_guesses)
        
        loss = wd(real_guesses, fake_guesses)
        
        loss.backward()
        dis_opt.step()
            
        losses.append(loss.item())
        real_accs.append(real_acc * 100)
        fake_accs.append(fake_acc * 100)
    
    loss        = (min(losses),    np.quantile(losses, .5),    max(losses))
    real_acc    = (min(real_accs), np.quantile(real_accs, .5), max(real_accs))
    fake_acc    = (min(fake_accs), np.quantile(fake_accs, .5), max(fake_accs))
    
    return(gen_loss.item(), loss, real_acc, fake_acc)
        





def train():
    
    try: shutil.rmtree("output") 
    except: pass 

    os.mkdir("output")
    os.mkdir("output/gens")
    torch.save(args, "output/args.pt")
    
    gen = Generator()
    gen.summary()

    dises = [Discriminator() for d in range(args.dises)]
    dises[0].summary()
    
    loss_acc = {
        "change_level"  : [], 
        "gen_loss"      : [],  
        "dis_loss"      : [], 
        "dis_real_acc"  : [],  
        "dis_fake_acc"  : []}
    
    real_nums = [i for i in range(0, 18, 2)]
    example_seeds = torch.normal(
        mean = torch.zeros([9, args.seed_size]),
        std  = torch.ones( [9, args.seed_size]))
    torch.save(example_seeds, "output/seeds.pt")
    
    total_epochs = 0
    manager = enlighten.Manager()
    TE = manager.counter(total = sum([v for v in args.epochs.values()]), desc = "Total:", unit = "ticks", color = "blue")
    
    for level in [0, 0.0, 1, 1.0, 2, 2.0, 3, 3.0, 4]:
        loss_acc["change_level"].append(total_epochs)
        epochs = args.epochs[floor(level), type(level)]
        batch_sizes = args.batch_sizes[floor(level), type(level)]
        gen_opt = Adam(gen.parameters(), args.gen_lr[floor(level), type(level)])
        dis_opts = [Adam(dis.parameters(), args.dis_lr[floor(level), type(level)]) for dis in dises]

        prev_batches = {i : [] for i, _ in enumerate(args.prev_batch)}
        E = manager.counter(total = epochs, desc = "Level {}:".format(level), unit = "ticks", color = "blue")
        if(type(level)) == int:
            buffer = get_buffer(2**(level+2))
            reals = [buffer[i] for i in real_nums]
        else:
            old_buffer = get_buffer(2**(floor(level+2)))
            old_buffer = F.interpolate(old_buffer.permute(0, 3, 1, 2), scale_factor = 2, mode = "nearest").permute(0, 2, 3, 1)
            new_buffer = get_buffer(2**(floor(level+3)))
        
        for e in range(1, epochs+1):
            if(type(level) == int): pass
            else:
                level += 1/(epochs+1)
                alpha = level - floor(level)
                buffer = (1 - alpha) * old_buffer + alpha * new_buffer
                reals = [buffer[i] for i in real_nums]
                
            gen.change_level(level)
            for dis in dises: dis.change_level(level)
            total_epochs += 1
            gen.epochs = total_epochs 
            for dis in dises: dis.epochs = total_epochs
            TE.update()
            E.update()
            
            gen_loss, dis_loss, real_acc, fake_acc = epoch(buffer, batch_sizes, prev_batches, gen, gen_opt, dises, dis_opts)
            loss_acc["gen_loss"].append(gen_loss)
            loss_acc["dis_loss"].append(dis_loss)
            loss_acc["dis_real_acc"].append(real_acc)
            loss_acc["dis_fake_acc"].append(fake_acc)
            
            plotting = e == 1 or e == epochs or e % args.plotting == 0
            showing  = e == 1 or e == epochs or e % (args.plotting * args.show_plots) == 0
            if(plotting):
                gen.eval()
                fakes = gen(example_seeds).detach().cpu()
                plots(loss_acc, reals, fakes, total_epochs, round(level, 3), buffer.shape[2], showing)
            keeping_gen = e == 1 or e == epochs or e % args.keep_gen == 0
            if(keeping_gen): 
                copy_gen = deepcopy(gen)
                copy_gen.eval()
                torch.save(copy_gen, "output/gens/gen_{}.pt".format(str(total_epochs).zfill(10)))
    
train()
print("\n\nDone training!\n\n")

make_training_vid()
make_seeding_vid(betweens = 5, fps = 10)
# %%
