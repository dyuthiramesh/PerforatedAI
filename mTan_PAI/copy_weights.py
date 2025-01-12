import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import csv
from perforatedai import pb_network as PN
from perforatedai import pb_globals as PBG

from random import SystemRandom
import modelsPAI as models
import modelsPAI2 as models2
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

PBG.inputDimensions = [-1, -1, 0]

PBG.modulesToConvert.append(models.mtan_time_embedder)
PBG.modulesToConvert.append(models.multiTimeAttention)
PBG.modulesToConvert.append(nn.GRU)
PBG.modluesWithProcessing.append(nn.GRU)
PBG.moduleProcessingClasses.append(GRUCellProcessor)
PBG.modulesToConvert.append(models.reverseGru)
PBG.modluesWithProcessing.append(models.reverseGru)
PBG.moduleProcessingClasses.append(ReverseGRUCellProcessor)

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
    
    model = fullModel(rec, dec, classifier)
    # print(model)
    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    print('Before loading parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    model = PN.loadPAIModel(model, 'best_model_pai.pt').to('cuda')
    model = torch.load("customPAImtan.pt")
    print("The old model used is: ",model)
    network = torch.load("SecondModel.pt")
    # network = PN.loadPAIModel(model, "finalPAIcustom.pt")

    print("The new network that is made is: ",network)
    
    for name, buffer in network.named_buffers():
        print(f"Buffer Name: {name}, Buffer Value: {buffer}")

    rec_old = model.rec
    rec_new = network.rec

    dec_old = model.dec
    dec_new = network.dec

    classifier_old = model.classifier
    classifier_new = network.classifier


    rec_new.att = rec_old.att
    rec_new.hiddens_to_z0 = rec_old.hiddens_to_z0

    rec_new.gru_rnn.layerArray[0].forward_gru.weight_ih = rec_old.gru_rnn.layerArray[0].weight_ih_l0
    rec_new.gru_rnn.layerArray[0].forward_gru.weight_hh = rec_old.gru_rnn.layerArray[0].weight_hh_l0
    rec_new.gru_rnn.layerArray[0].forward_gru.bias_ih = rec_old.gru_rnn.layerArray[0].bias_ih_l0
    rec_new.gru_rnn.layerArray[0].forward_gru.bias_hh = rec_old.gru_rnn.layerArray[0].bias_hh_l0

    rec_new.gru_rnn.layerArray[0].backward_gru.weight_ih = rec_old.gru_rnn.layerArray[0].weight_ih_l0_reverse
    rec_new.gru_rnn.layerArray[0].backward_gru.weight_hh = rec_old.gru_rnn.layerArray[0].weight_hh_l0_reverse
    rec_new.gru_rnn.layerArray[0].backward_gru.bias_ih = rec_old.gru_rnn.layerArray[0].bias_ih_l0_reverse
    rec_new.gru_rnn.layerArray[0].backward_gru.bias_hh = rec_old.gru_rnn.layerArray[0].bias_hh_l0_reverse

    rec_new.gru_rnn.layerArray[1].forward_gru.weight_ih = rec_old.gru_rnn.layerArray[1].weight_ih_l0
    rec_new.gru_rnn.layerArray[1].forward_gru.weight_hh = rec_old.gru_rnn.layerArray[1].weight_hh_l0
    rec_new.gru_rnn.layerArray[1].forward_gru.bias_ih = rec_old.gru_rnn.layerArray[1].bias_ih_l0
    rec_new.gru_rnn.layerArray[1].forward_gru.bias_hh = rec_old.gru_rnn.layerArray[1].bias_hh_l0

    rec_new.gru_rnn.layerArray[1].backward_gru.weight_ih = rec_old.gru_rnn.layerArray[1].weight_ih_l0_reverse
    rec_new.gru_rnn.layerArray[1].backward_gru.weight_hh = rec_old.gru_rnn.layerArray[1].weight_hh_l0_reverse
    rec_new.gru_rnn.layerArray[1].backward_gru.bias_ih = rec_old.gru_rnn.layerArray[1].bias_ih_l0_reverse
    rec_new.gru_rnn.layerArray[1].backward_gru.bias_hh = rec_old.gru_rnn.layerArray[1].bias_hh_l0_reverse

    rec_new.gru_rnn.layerArray[2].forward_gru.weight_ih = rec_old.gru_rnn.layerArray[2].weight_ih_l0
    rec_new.gru_rnn.layerArray[2].forward_gru.weight_hh = rec_old.gru_rnn.layerArray[2].weight_hh_l0
    rec_new.gru_rnn.layerArray[2].forward_gru.bias_ih = rec_old.gru_rnn.layerArray[2].bias_ih_l0
    rec_new.gru_rnn.layerArray[2].forward_gru.bias_hh = rec_old.gru_rnn.layerArray[2].bias_hh_l0

    rec_new.gru_rnn.layerArray[2].backward_gru.weight_ih = rec_old.gru_rnn.layerArray[2].weight_ih_l0_reverse
    rec_new.gru_rnn.layerArray[2].backward_gru.weight_hh = rec_old.gru_rnn.layerArray[2].weight_hh_l0_reverse
    rec_new.gru_rnn.layerArray[2].backward_gru.bias_ih = rec_old.gru_rnn.layerArray[2].bias_ih_l0_reverse
    rec_new.gru_rnn.layerArray[2].backward_gru.bias_hh = rec_old.gru_rnn.layerArray[2].bias_hh_l0_reverse

    rec_new.gru_rnn.layerArray[3].forward_gru.weight_ih = rec_old.gru_rnn.layerArray[3].weight_ih_l0
    rec_new.gru_rnn.layerArray[3].forward_gru.weight_hh = rec_old.gru_rnn.layerArray[3].weight_hh_l0
    rec_new.gru_rnn.layerArray[3].forward_gru.bias_ih = rec_old.gru_rnn.layerArray[3].bias_ih_l0
    rec_new.gru_rnn.layerArray[3].forward_gru.bias_hh = rec_old.gru_rnn.layerArray[3].bias_hh_l0

    rec_new.gru_rnn.layerArray[3].backward_gru.weight_ih = rec_old.gru_rnn.layerArray[3].weight_ih_l0_reverse
    rec_new.gru_rnn.layerArray[3].backward_gru.weight_hh = rec_old.gru_rnn.layerArray[3].weight_hh_l0_reverse
    rec_new.gru_rnn.layerArray[3].backward_gru.bias_ih = rec_old.gru_rnn.layerArray[3].bias_ih_l0_reverse
    rec_new.gru_rnn.layerArray[3].backward_gru.bias_hh = rec_old.gru_rnn.layerArray[3].bias_hh_l0_reverse

    

    
    rec_new.embedder1 = rec_old.embedder1
    rec_new.embedder2 = rec_old.embedder2

    dec_new.att = dec_old.att
    dec_new.z0_to_obs = dec_old.z0_to_obs

    dec_new.gru_rnn.layerArray[0].forward_gru.weight_ih = dec_old.gru_rnn.layerArray[0].weight_ih_l0
    dec_new.gru_rnn.layerArray[0].forward_gru.weight_hh = dec_old.gru_rnn.layerArray[0].weight_hh_l0
    dec_new.gru_rnn.layerArray[0].forward_gru.bias_ih = dec_old.gru_rnn.layerArray[0].bias_ih_l0
    dec_new.gru_rnn.layerArray[0].forward_gru.bias_hh = dec_old.gru_rnn.layerArray[0].bias_hh_l0

    dec_new.gru_rnn.layerArray[0].backward_gru.weight_ih = dec_old.gru_rnn.layerArray[0].weight_ih_l0_reverse
    dec_new.gru_rnn.layerArray[0].backward_gru.weight_hh = dec_old.gru_rnn.layerArray[0].weight_hh_l0_reverse
    dec_new.gru_rnn.layerArray[0].backward_gru.bias_ih = dec_old.gru_rnn.layerArray[0].bias_ih_l0_reverse
    dec_new.gru_rnn.layerArray[0].backward_gru.bias_hh = dec_old.gru_rnn.layerArray[0].bias_hh_l0_reverse
    
    dec_new.gru_rnn.layerArray[1].forward_gru.weight_ih = dec_old.gru_rnn.layerArray[1].weight_ih_l0
    dec_new.gru_rnn.layerArray[1].forward_gru.weight_hh = dec_old.gru_rnn.layerArray[1].weight_hh_l0
    dec_new.gru_rnn.layerArray[1].forward_gru.bias_ih = dec_old.gru_rnn.layerArray[1].bias_ih_l0
    dec_new.gru_rnn.layerArray[1].forward_gru.bias_hh = dec_old.gru_rnn.layerArray[1].bias_hh_l0
    
    dec_new.gru_rnn.layerArray[1].backward_gru.weight_ih = dec_old.gru_rnn.layerArray[1].weight_ih_l0_reverse
    dec_new.gru_rnn.layerArray[1].backward_gru.weight_hh = dec_old.gru_rnn.layerArray[1].weight_hh_l0_reverse
    dec_new.gru_rnn.layerArray[1].backward_gru.bias_ih = dec_old.gru_rnn.layerArray[1].bias_ih_l0_reverse
    dec_new.gru_rnn.layerArray[1].backward_gru.bias_hh = dec_old.gru_rnn.layerArray[1].bias_hh_l0_reverse
    
    dec_new.gru_rnn.layerArray[2].forward_gru.weight_ih = dec_old.gru_rnn.layerArray[2].weight_ih_l0
    dec_new.gru_rnn.layerArray[2].forward_gru.weight_hh = dec_old.gru_rnn.layerArray[2].weight_hh_l0
    dec_new.gru_rnn.layerArray[2].forward_gru.bias_ih = dec_old.gru_rnn.layerArray[2].bias_ih_l0
    dec_new.gru_rnn.layerArray[2].forward_gru.bias_hh = dec_old.gru_rnn.layerArray[2].bias_hh_l0
    
    dec_new.gru_rnn.layerArray[2].backward_gru.weight_ih = dec_old.gru_rnn.layerArray[2].weight_ih_l0_reverse
    dec_new.gru_rnn.layerArray[2].backward_gru.weight_hh = dec_old.gru_rnn.layerArray[2].weight_hh_l0_reverse
    dec_new.gru_rnn.layerArray[2].backward_gru.bias_ih = dec_old.gru_rnn.layerArray[2].bias_ih_l0_reverse
    dec_new.gru_rnn.layerArray[2].backward_gru.bias_hh = dec_old.gru_rnn.layerArray[2].bias_hh_l0_reverse
    
    dec_new.gru_rnn.layerArray[3].forward_gru.weight_ih = dec_old.gru_rnn.layerArray[3].weight_ih_l0
    dec_new.gru_rnn.layerArray[3].forward_gru.weight_hh = dec_old.gru_rnn.layerArray[3].weight_hh_l0
    dec_new.gru_rnn.layerArray[3].forward_gru.bias_ih = dec_old.gru_rnn.layerArray[3].bias_ih_l0
    dec_new.gru_rnn.layerArray[3].forward_gru.bias_hh = dec_old.gru_rnn.layerArray[3].bias_hh_l0
    
    dec_new.gru_rnn.layerArray[3].backward_gru.weight_ih = dec_old.gru_rnn.layerArray[3].weight_ih_l0_reverse
    dec_new.gru_rnn.layerArray[3].backward_gru.weight_hh = dec_old.gru_rnn.layerArray[3].weight_hh_l0_reverse
    dec_new.gru_rnn.layerArray[3].backward_gru.bias_ih = dec_old.gru_rnn.layerArray[3].bias_ih_l0_reverse
    dec_new.gru_rnn.layerArray[3].backward_gru.bias_hh = dec_old.gru_rnn.layerArray[3].bias_hh_l0_reverse

    dec_new.embedder1 = dec_old.embedder1
    dec_new.embedder2 = dec_old.embedder2

    classifier_new.classifier = classifier_old.classifier
    
    classifier_new.gru_rnn.layerArray[0].gru_cell.weight_ih = classifier_old.gru_rnn.layerArray[0].weight_ih_l0
    classifier_new.gru_rnn.layerArray[0].gru_cell.weight_hh = classifier_old.gru_rnn.layerArray[0].weight_hh_l0
    classifier_new.gru_rnn.layerArray[0].gru_cell.bias_ih = classifier_old.gru_rnn.layerArray[0].bias_ih_l0
    classifier_new.gru_rnn.layerArray[0].gru_cell.bias_hh = classifier_old.gru_rnn.layerArray[0].bias_hh_l0
    
    classifier_new.gru_rnn.layerArray[1].gru_cell.weight_ih = classifier_old.gru_rnn.layerArray[1].weight_ih_l0
    classifier_new.gru_rnn.layerArray[1].gru_cell.weight_hh = classifier_old.gru_rnn.layerArray[1].weight_hh_l0
    classifier_new.gru_rnn.layerArray[1].gru_cell.bias_ih = classifier_old.gru_rnn.layerArray[1].bias_ih_l0
    classifier_new.gru_rnn.layerArray[1].gru_cell.bias_hh = classifier_old.gru_rnn.layerArray[1].bias_hh_l0
    
    classifier_new.gru_rnn.layerArray[2].gru_cell.weight_ih = classifier_old.gru_rnn.layerArray[2].weight_ih_l0
    classifier_new.gru_rnn.layerArray[2].gru_cell.weight_hh = classifier_old.gru_rnn.layerArray[2].weight_hh_l0
    classifier_new.gru_rnn.layerArray[2].gru_cell.bias_ih = classifier_old.gru_rnn.layerArray[2].bias_ih_l0
    classifier_new.gru_rnn.layerArray[2].gru_cell.bias_hh = classifier_old.gru_rnn.layerArray[2].bias_hh_l0
    
    classifier_new.gru_rnn.layerArray[3].gru_cell.weight_ih = classifier_old.gru_rnn.layerArray[3].weight_ih_l0
    classifier_new.gru_rnn.layerArray[3].gru_cell.weight_hh = classifier_old.gru_rnn.layerArray[3].weight_hh_l0
    classifier_new.gru_rnn.layerArray[3].gru_cell.bias_ih = classifier_old.gru_rnn.layerArray[3].bias_ih_l0
    classifier_new.gru_rnn.layerArray[3].gru_cell.bias_hh = classifier_old.gru_rnn.layerArray[3].bias_hh_l0

    network.rec = rec_new
    network.dec = dec_new
    network.classifier = classifier_new
   
    # for name, layer in model.named_parameters():
    #     print(name)
        # print(layer)

    # for name, layer in network.named_parameters():
    #     print(name)
        # print(layer)

    # with open('weights_comparison.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Layer Name", "Source Weights"])

    #     for name1, layer1 in model.named_parameters():
        
    #         source_weights = layer1.detach().cpu().numpy()
            
    #         writer.writerow([
    #             name1, 
    #             source_weights.tolist(),
    #         ])
    # with open('weights_comparison2.csv', mode='w', newline='') as file:
    #     writer2 = csv.writer(file)
    #     writer2.writerow(["Layer Name", "Source Weights"])
    
    #     for name2, layer2 in network.named_parameters():
        
    #         target_weights = layer2.detach().cpu().numpy()
                    
    #         writer2.writerow([
    #             name2, 
    #             target_weights.tolist(), 
    #         ])
        
    # print("Weights comparison saved to 'weights_comparison.csv'.")
    for name, buffer in network.named_buffers():
        print(f"Buffer Name: {name}, Buffer Value: {buffer}")

    torch.save(network, "SecondCopiedModel.pt")
    # print("Size of the weights are: ",(rec_gru.weight_ih_l0_reverse.shape))
    # print(type(rec_gru))
    # for name, model in rec_gru.named_parameters():
    #     print(name)