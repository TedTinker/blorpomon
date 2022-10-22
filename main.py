#%%
import enlighten 
from random import sample

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torchgan.losses import WassersteinGeneratorLoss as WG
from torchgan.losses import WassersteinDiscriminatorLoss as WD

from utils import args, get_buffer, get_batch, plot_losses, plot_examples, make_vid
from generator import Generator
from discriminator import Discriminator


    
def epoch(test, buffer, batch_size, gen, gen_opt, dis, dis_opt):
    
    if(test): gen.eval() ; dis.eval() 
    else:     gen.train(); dis.train() 
    
    wg = WG()
    
    gen_opt.zero_grad()
    fake_batch = gen.go(batch_size)
    guesses = dis(fake_batch, None)
    #gen_loss = F.binary_cross_entropy(guesses, torch.ones(guesses.shape))
    gen_loss = wg(guesses)
    
    if(not test):
        gen_loss.backward()
        gen_opt.step()
        
    wd = WD()
    
    real_batch = get_batch(buffer, batch_size)
    with torch.no_grad(): 
        fake_batch = gen.go(batch_size)
    
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
    
    return(gen_loss.item(), loss.item(), real_acc*100, fake_acc*100)
        


def train_step(e, buffer, batch_size, testing, plotting, loss_acc, reals, example_seeds, gen, gen_opt, dis, dis_opt):
            
    gen_loss, dis_loss, real_acc, fake_acc = epoch(False, buffer, batch_size, gen, gen_opt, dis, dis_opt)
    loss_acc["gen_train_loss"].append(gen_loss)
    loss_acc["dis_train_loss"].append(dis_loss)
    loss_acc["dis_real_train_acc"].append(real_acc)
    loss_acc["dis_fake_train_acc"].append(fake_acc)
    
    if(testing):
        loss_acc["test_xs"].append(e)
        gen_loss, dis_loss, real_acc, fake_acc = epoch(True, buffer, batch_size, gen, gen_opt, dis, dis_opt)
        loss_acc["gen_test_loss" ].append(gen_loss)
        loss_acc["dis_test_loss"].append(dis_loss)
        loss_acc["dis_real_test_acc"].append(real_acc)
        loss_acc["dis_fake_test_acc"].append(fake_acc)
        
    if(plotting):
        gen.eval()
        fakes = gen(example_seeds).detach().cpu()
        plot_examples(reals, fakes, e)
        plot_losses(loss_acc)
    


def train():
    
    buffer = get_buffer()
    
    gen = Generator()
    gen_opt = Adam(gen.parameters(), args.gen_lr)

    dis = Discriminator()
    dis_opt = Adam(dis.parameters(), args.dis_lr)
    
    example_seeds = torch.normal(
        mean = torch.zeros([9, args.seed_size]),
        std  = torch.ones( [9, args.seed_size]))
    
    loss_acc = {
        "change_level" : [],        "test_xs" : [],
        "gen_train_loss" : [],      "gen_test_loss" : [],
        "dis_train_loss" : [],      "dis_test_loss" : [],
        "dis_real_train_acc" : [],  "dis_real_test_acc" : [],
        "dis_fake_train_acc" : [],  "dis_fake_test_acc" : []}

    total_epochs = 0
    
    reals = [buffer[i] for i in range(0, 18, 2)]
    manager = enlighten.Manager()
    E = manager.counter(total = args.epochs, desc = "Epochs:", unit = "ticks", color = "blue")
    for e in range(args.epochs):
        total_epochs += 1
        E.update()
        train_step(
            total_epochs, buffer, args.batch_size, 
            e == 0 or e == args.epochs-1 or (e+1) % args.testing == 0,
            e == 0 or e == args.epochs-1 or (e+1) % args.plotting == 0, 
            loss_acc, reals, example_seeds, gen, gen_opt, dis, dis_opt)
                
    make_vid()
        
train()
# %%
