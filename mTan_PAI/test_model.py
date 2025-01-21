import torch
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from perforatedai import pb_network as PN
from perforatedai import pb_globals as PBG

from random import SystemRandom
import modelsPAI2 as models
import utils
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--justTest', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.1, 
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true', 
                    help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--alpha', type=int, default=100.)
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--multiplier', type=float, default=1)

args = parser.parse_args()


class GRUCellProcessor():
    #Post processing does eventually need to return h_t and c__t, but h_t gets modified py the PB
    #nodes first so it needs to be extracted in post 1, and then gets added back in post 2
    def post_n1(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        #store the cell state temporarily and just use the hidden state to do PB functions
        self.c_t_n = c_t
        return h_t
    def post_n2(self, *args, **kawrgs):
        h_t = args[0]
        return h_t, self.c_t_n
    #these Grus are just getting passed input and no hidden state for some reason so just pass it along
    def pre_d(self, *args, **kwargs):
        return args, kwargs
        
    #for post processsing its just getting passed the output, which is (h_t,c_t). Then it wants to just pass along h_t as the output for the function to be passed to the parent while retaining both
    def post_d(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return h_t
    def clear_processor(self):
        for var in ['c_t_n', 'h_t_d', 'c_t_d']:
            if(not getattr(self,var,None) is None):
                delattr(self,var)
    
#the classifier gru is passing along the cell state instead of the hidden state so use that isntead
class ReverseGRUCellProcessor():
    #Post processing does eventually need to return h_t and c__t, but c_t gets modified py the PB
    #nodes first so it needs to be extracted in post 1, and then gets added back in post 2
    def post_n1(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        #store the cell state temporarily and just use the hidden state to do PB functions
        self.h_t_n = h_t
        return c_t
    def post_n2(self, *args, **kawrgs):
        c_t = args[0]
        return self.h_t_n, c_t
    #these Grus are just getting passed input and no hidden state for some reason so just pass it along
    def pre_d(self, *args, **kwargs):
        return args, kwargs
        
    #for post processsing its just getting passed the output, which is (h_t,c_t). Then it wants to just pass along h_t as the output for the function to be passed to the parent while retaining both
    def post_d(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return c_t
    def clear_processor(self):
        for var in ['h_t_n', 'h_t_d', 'c_t_d']:
            if(not getattr(self,var,None) is None):
                delattr(self,var)

class fullModel(nn.Module):
    def __init__(self, rec, dec, classifier):
        super(fullModel, self).__init__()
        self.rec = rec
        self.dec = dec
        self.classifier = classifier
        
    def forward(self, observed_data, observed_mask, observed_tp):
        #import pdb; pdb.set_trace()
        out = self.rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
        qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
        epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        pred_y = self.classifier(z0)
        pred_x = self.dec(
            z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
        pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, seqlen, dim
        return pred_x, pred_y, qz0_mean, qz0_logvar

if __name__ == '__main__':
    experiment_id = int(SystemRandom().random()*100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args)
    
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    
    internal = int(100 * args.multiplier)
    nlin = int(50 * args.multiplier)
    embed_time = 128
    embed_time = int(embed_time * args.multiplier)
    args.latent_dim = int(args.latent_dim * args.multiplier)
    args.rec_hidden = int(args.rec_hidden * args.multiplier)
    args.gen_hidden = int(args.gen_hidden * args.multiplier)
        
    
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., 128), args.latent_dim, args.rec_hidden, nlin, 128 , learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
            embed_time=embed_time, learn_emb=args.learn_emb, num_heads=args.enc_num_heads, device=device).to(device)

    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., 128), args.latent_dim, args.gen_hidden, nlin, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=embed_time, learn_emb=args.learn_emb, num_heads=args.dec_num_heads, device=device).to(device)
    classifier = models.create_classifier(args.latent_dim, internal, args.rec_hidden).to(device)
    
    # model = fullModel(rec, dec, classifier)
    model = torch.load("CheckingModelCopied.pt")
    # model = PN.loadPAIModel(model, 'best_model_pai.pt').to('cuda')
    # model = PN.loadPAIModel(model, 'best_model_pai.pt').to('cuda')
    
    print(model)
    for name, buffer in model.named_buffers():
        print(f"Buffer Name: {name}, Buffer Value: {buffer}")
    
    rec=model.rec
    dec = model.dec
    classifier = model.classifier
    for item in [rec,dec,classifier]:
        for name, param in item.named_parameters():
            print(name)
            print(param)

    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    
    if(args.justTest):
        model.eval()
        test_loss, test_acc, test_auc = utils.evaluate_classifier(
            model.rec, test_loader, args=args, classifier=model.classifier, reconst=True, num_sample=1, dim=dim, device=device)
        print('test_acc')
        print(test_acc)
        print('test_auc')
        print(test_auc)