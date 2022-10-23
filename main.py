#%%
import os
import enlighten 
from random import sample
from copy import deepcopy

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torchgan.losses import WassersteinGeneratorLoss as WG
from torchgan.losses import WassersteinDiscriminatorLoss as WD

from utils import args, get_buffer, get_batch, plot_losses, plot_examples, make_training_vid, make_seeding_vid
from generator import Generator
from discriminator import Discriminator


    
def epoch(test, buffer, batch_size, gen, gen_opt, dises, dis_opts):
    
    if(test): 
        gen.eval()
        for dis in dises:
            dis.eval()
    else:     
        gen.train()
        for dis in dises:
            dis.train()
    
    wg = WG()
    
    gen_opt.zero_grad()
    fake_batch = gen.go(batch_size)
    guesses = torch.cat([dis(fake_batch, None) for dis in dises], dim = 1)
    gen_loss = wg(guesses)
    
    if(not test):
        gen_loss.backward()
        gen_opt.step()
        
    wd = WD()
    
    real_batch = get_batch(buffer, batch_size)
    with torch.no_grad(): 
        fake_batch = gen.go(batch_size)
        
    losses = [] ; real_accs = [] ; fake_accs = []
        
    for dis, dis_opt in zip(dises, dis_opts):
    
        dis_opt.zero_grad()
        real_guesses = dis(real_batch, True, test)
        real_guesses_l = [round(g[0]) for g in real_guesses.tolist()]
        real_acc = sum([1 if g == 1 else 0 for g in real_guesses_l]) / len(real_guesses)
        
        dis_opt.zero_grad()
        fake_guesses = dis(fake_batch, False, test)
        fake_guesses_l = [round(g[0]) for g in fake_guesses.tolist()]
        fake_acc = sum([1 if g == 0 else 0 for g in fake_guesses_l]) / len(fake_guesses)
        
        loss = wd(real_guesses, fake_guesses)
        
        if(not test):
            loss.backward()
            dis_opt.step()
            
        losses.append(loss.item())
        real_accs.append(real_acc * 100)
        fake_accs.append(fake_acc * 100)
    
    return(gen_loss.item(), losses, real_accs, fake_accs)
        


def train_step(e, buffer, batch_size, testing, loss_acc, gen, gen_opt, dises, dis_opts):
            
    gen_loss, dis_losses, real_accs, fake_accs = epoch(False, buffer, batch_size, gen, gen_opt, dises, dis_opts)
    loss_acc["gen_train_loss"].append(gen_loss)
    for d in range(len(dis_losses)):
        loss_acc["dis_train_loss"][d].append(dis_losses[d])
        loss_acc["dis_real_train_acc"][d].append(real_accs[d])
        loss_acc["dis_fake_train_acc"][d].append(fake_accs[d])
    
    if(testing):
        loss_acc["test_xs"].append(e)
        gen_loss, dis_losses, real_accs, fake_accs = epoch(True, buffer, batch_size, gen, gen_opt, dises, dis_opts)
        loss_acc["gen_test_loss" ].append(gen_loss)
        for d in range(len(dis_losses)):
            loss_acc["dis_test_loss"][d].append(dis_losses[d])
            loss_acc["dis_real_test_acc"][d].append(real_accs[d])
            loss_acc["dis_fake_test_acc"][d].append(fake_accs[d])
    


def train():
    
    buffer = get_buffer()
    
    gen = Generator()
    gen_opt = Adam(gen.parameters(), args.gen_lr)
    gens = []

    dises = [Discriminator() for d in range(args.dises)]
    dis_opts = [Adam(dis.parameters(), args.dis_lr) for dis in dises]
    
    example_seeds = torch.normal(
        mean = torch.zeros([9, args.seed_size]),
        std  = torch.ones( [9, args.seed_size]))
    
    loss_acc = {
        "change_level" : [],   "test_xs" : [],
        "gen_train_loss" : [], "gen_test_loss" : [],
        "dis_train_loss" : [[] for d in range(args.dises)],      "dis_test_loss" : [[] for d in range(args.dises)],
        "dis_real_train_acc" : [[] for d in range(args.dises)],  "dis_real_test_acc" : [[] for d in range(args.dises)],
        "dis_fake_train_acc" : [[] for d in range(args.dises)],  "dis_fake_test_acc" : [[] for d in range(args.dises)]}

    total_epochs = 0
    
    reals = [buffer[i] for i in range(0, 18, 2)]
    manager = enlighten.Manager()
    E = manager.counter(total = args.epochs, desc = "Epochs:", unit = "ticks", color = "blue")
    for e in range(1, args.epochs+1):
        total_epochs += 1
        E.update()
        train_step(
            total_epochs, buffer, args.batch_size, 
            e == 0 or e == args.epochs-1 or (e+1) % args.testing == 0,
            loss_acc, gen, gen_opt, dises, dis_opts)
        
        plotting = e == 1 or e == args.epochs or e % args.plotting == 0
        showing  = e == 1 or e == args.epochs or e % (args.plotting * args.show_plots) == 0
        if(plotting):
            gen.eval()
            fakes = gen(example_seeds).detach().cpu()
            plot_examples(reals, fakes, e, showing)
            plot_losses(loss_acc, showing)
        keeping_gen = e == 1 or e == args.epochs or e % args.keep_gen == 0
        if(keeping_gen):
            gens.append((e, deepcopy(gen)))
                
    make_training_vid()
    make_seeding_vid(gens, example_seeds, betweens = 25, fps = 10)
    
train()
print("\n\nDone!\n\n")
# %%
