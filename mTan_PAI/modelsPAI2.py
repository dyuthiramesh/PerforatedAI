import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gru import StandardGRU, BidirectionalGRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class reverseGru(StandardGRU):
    def __init__(self, *args, **kwargs):
        super(reverseGru, self).__init__(*args, **kwargs)


class create_classifier(nn.Module):
 
    def __init__(self, latent_dim, internal = 300, nhidden=16, N=2):
        super(create_classifier, self).__init__()
        print(latent_dim,nhidden)
        self.gru_rnn = reverseGru(latent_dim, nhidden)
        self.numberHidden = nhidden
        self.N = N
        self.classifier = nn.Sequential(
            nn.Linear(self.numberHidden, internal),
            nn.ReLU(),
            nn.Linear(internal, internal),
            nn.ReLU(),
            nn.Linear(internal, N))
    def _initialize_numberHidden(self, n):
        self.numberHidden = n
    
    def _initialize_classifier(self):
        """Reinitialize the classifier layer."""
        original_weights = self.classifier[0].weight.data.clone()
        original_biases = self.classifier[0].bias.data.clone()
        # print("Original Weights size: ",original_weights.shape)
        # print("Original Bias size: ",original_biases.shape)
        self.classifier = nn.Sequential(
            nn.Linear(self.numberHidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, self.N) 
        ).to(device)

        original_weights = self.classifier[0].weight.data.clone()
        original_biases = self.classifier[0].bias.data.clone()
        # print("Original Weights size: ",original_weights.shape)
        # print("Original Bias size: ",original_biases.shape)
    
    def prune_classifier(self, threshold=0.04):
        for i in range(len(self.classifier[0].layerArray)):

            original_weights_1 = self.classifier[0].layerArray[i].weight.data.clone()
            original_biases_1 = self.classifier[0].layerArray[i].bias.data.clone()
            l1_norms_1 = torch.sum(torch.abs(original_weights_1), dim=1)  
            # print(f"First layer shape of l1 norms: ",l1_norms_1.shape)
            top_k_1 = int((1 - threshold) * l1_norms_1.size(0))
            _, top_indices_1 = torch.topk(l1_norms_1, top_k_1)
            
            new_layer_1 = nn.Linear(self.numberHidden, top_k_1).to(device)
            new_layer_1.weight.data = original_weights_1[top_indices_1, :]
            new_layer_1.bias.data = original_biases_1[top_indices_1]
            self.classifier[0].layerArray[i] = new_layer_1
            
            original_weights_2 = self.classifier[2].layerArray[i].weight.data.clone()
            original_biases_2 = self.classifier[2].layerArray[i].bias.data.clone()
            
            l1_norms_2 = torch.sum(torch.abs(original_weights_2), dim=1) 
            top_k_2_out = int((1 - threshold) * l1_norms_2.size(0))
            _, top_indices_2_out = torch.topk(l1_norms_2, top_k_2_out)
            
            l1_norms_2_in = torch.sum(torch.abs(original_weights_2), dim=0)
            top_k_2_in = int((1 - threshold) * l1_norms_2_in.size(0))
            _, top_indices_2_in = torch.topk(l1_norms_2_in, top_k_2_in)
            
            new_layer_2 = nn.Linear(top_k_2_in, top_k_2_out).to(device)
            new_layer_2.weight.data = original_weights_2[top_indices_2_out][:, top_indices_2_in]
            new_layer_2.bias.data = original_biases_2[top_indices_2_out]
            self.classifier[2].layerArray[i] = new_layer_2
            
            original_weights_3 = self.classifier[4].layerArray[i].weight.data.clone()
            original_biases_3 = self.classifier[4].layerArray[i].bias.data.clone()
            
            l1_norms_3 = torch.sum(torch.abs(original_weights_3), dim=0)  
            # print(f"Third layer l1 norms shape: ",l1_norms_3.shape)
            top_k_3_in = int((1 - threshold) * l1_norms_3.size(0))
            _, top_indices_3_in = torch.topk(l1_norms_3, top_k_3_in)
            
            new_layer_3 = nn.Linear(top_k_3_in, self.N).to(device)
            new_layer_3.weight.data = original_weights_3[:, top_indices_3_in]
            new_layer_3.bias.data = original_biases_3
            self.classifier[4].layerArray[i] = new_layer_3
        
    def update_input_size(self, new_input_size):

        
        for i in range(len(self.classifier[0].layerArray)):
            print("============================",self.classifier[0].layerArray[i])
            original_weights = self.classifier[0].layerArray[i].weight.data.clone()
            
            original_biases = self.classifier[0].layerArray[i].bias.data.clone()

            # original_biases = original_biases[0].layerArray[i].bias.data.clone()     
        
            l1_norms = torch.sum(torch.abs(original_weights), dim=0)   
        
            _, top_indices = torch.topk(l1_norms, new_input_size)      
        
            self.numberHidden = new_input_size
        
            new_input_layer = nn.Linear(self.numberHidden, 12)
        
            new_input_layer.weight.data = original_weights[:, top_indices] 
            new_input_layer.bias.data = original_biases  
            
            
            self.classifier[0].layerArray[i] = new_input_layer

        self.prune_classifier()
        
    def forward(self, z):
        _, out = self.gru_rnn(z)
        return self.classifier(out)
    

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.top_k_ratio = 0.96
        self.temp = 0
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    def reinitialize_dim(self, d):
        # print(" HAPPENING NOW ----------------------------------------------")
        self.dim = d
        new_input_to_last = self.dim * self.h

       
        original_weights = self.linears[-1].weight.data.clone()     
        original_biases = self.linears[-1].bias.data.clone() 

        l1_norms = torch.sum(torch.abs(original_weights), dim=0)
        _, top_indices = torch.topk(l1_norms, self.dim*self.h) 

        new_layer = nn.Linear(self.dim*self.h, self.nhidden)

        new_layer.weight.data = original_weights[:,top_indices]
        new_layer.bias.data = original_biases

        self.linears[-1] = new_layer     

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        
        # query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
        #               for l, x in zip(self.linears, (query, key))]
        
        query_layer = self.linears[0](query)  
        key_layer = self.linears[1](key)     

        query_reshaped = query_layer.view(query.size(0), -1, self.h, self.embed_time_k)
        key_reshaped = key_layer.view(key.size(0), -1, self.h, self.embed_time_k)

        query_transposed = query_reshaped.transpose(1, 2)
        key_transposed = key_reshaped.transpose(1, 2)
       
        query, key = query_transposed, key_transposed
        
        x, _ = self.attention(query, key, value, mask, dropout)
        

        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        
        output = self.linears[-1](x)
        
        return output
    
    

    
class mtan_time_embedder(nn.Module):
    def __init__(self, device, embed_time):
        super(mtan_time_embedder, self).__init__()
        self.device = device
        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)
        
    def forward(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    
class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, nlin = 50,
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = BidirectionalGRU(nhidden, nhidden)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, nlin),
            nn.ReLU(),
            nn.Linear(nlin, latent_dim * 2))
        if learn_emb:
            self.embedder1 = mtan_time_embedder(self.device, embed_time)
            self.embedder2 = mtan_time_embedder(self.device, embed_time)
    
    def _initialize_nhidden(self, n):
        self.nhidden = n

    def _initialize_hiddens_to_z0(self):
        """Reinitialize the hiddens_to_z0 layer."""
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * self.nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, 20 * 2)
        ).to(device)
    def update_input_size(self, new_input_size):
        
        print("Encoder hidden layers in model: ",self.hiddens_to_z0)
        for i in range(len(self.hiddens_to_z0[0].layerArray)):
            original_weights = self.hiddens_to_z0[0].layerArray[i].weight.data.clone()  
            original_biases = self.hiddens_to_z0[0].layerArray[i].bias.data.clone()     
        
            l1_norms = torch.sum(torch.abs(original_weights), dim=0)   
        
            _, top_indices = torch.topk(l1_norms, 2*new_input_size)      
        
            self.nhidden = new_input_size
        
            new_input_layer = nn.Linear(2 * self.nhidden, 50)
        
            new_input_layer.weight.data = original_weights[:, top_indices] 
            new_input_layer.bias.data = original_biases  
            
            self.hiddens_to_z0[0].layerArray[i] = new_input_layer
        
        self.prune_encoder_layers()
        print(self.hiddens_to_z0)


    def prune_encoder_layers(self, threshold=0.05):

        for i in range(len(self.hiddens_to_z0[0].layerArray)):
            original_weights_1 = self.hiddens_to_z0[0].layerArray[i].weight.data.clone()
            original_biases_1 = self.hiddens_to_z0[0].layerArray[i].bias.data.clone()
            l1_norms_1 = torch.sum(torch.abs(original_weights_1), dim=1)
            top_k_1 = int((1 - threshold) * l1_norms_1.size(0))
            _, top_indices_1 = torch.topk(l1_norms_1, top_k_1)
            
            new_layer_1 = nn.Linear(original_weights_1.size(1), top_k_1).to(self.device)
            new_layer_1.weight.data = original_weights_1[top_indices_1, :]
            new_layer_1.bias.data = original_biases_1[top_indices_1]
            self.hiddens_to_z0[0].layerArray[i] = new_layer_1

            original_weights_2 = self.hiddens_to_z0[2].layerArray[i].weight.data.clone()
            original_biases_2 = self.hiddens_to_z0[2].layerArray[i].bias.data.clone()
            l1_norms_2 = torch.sum(torch.abs(original_weights_2), dim=0)
            top_k_2 = int((1 - threshold) * l1_norms_2.size(0))
            _, top_indices_2 = torch.topk(l1_norms_2, top_k_2)
            
            new_layer_2 = nn.Linear(top_k_2, original_weights_2.size(0)).to(self.device)
            new_layer_2.weight.data = original_weights_2[:, top_indices_2]
            new_layer_2.bias.data = original_biases_2
            self.hiddens_to_z0[2].layerArray[i] = new_layer_2
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, time_steps):
        #time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        self.query = self.query.to(self.device)
        if self.learn_emb:
            key = self.embedder1(time_steps).to(self.device)
            query = self.embedder2(self.query.unsqueeze(0)).to(self.device)
            
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        
        out = self.att(query, key, x, mask)
       
        out, _ = self.gru_rnn(out)
       
        out = self.hiddens_to_z0(out)
        return out
    
   

   
class dec_mtan_rnn(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, nlin=50, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = BidirectionalGRU(latent_dim, nhidden)    
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, nlin),
            nn.ReLU(),
            nn.Linear(nlin, input_dim))
        if learn_emb:
            self.embedder1 = mtan_time_embedder(self.device, embed_time)
            self.embedder2 = mtan_time_embedder(self.device, embed_time)
        
    def _initialize_nhidden(self, n):
        # print("Hidden in decoder values is: ",self.nhidden)
        self.nhidden = n

    def _initialize_z0_to_obs(self):
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*self.nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, self.dim))

    def update_input_size(self, new_input_size):
       
        for i in range(len(self.z0_to_obs[0].layerArray)):

            original_weights = self.z0_to_obs[0].layerArray[i].weight.data.clone()  
            original_biases = self.z0_to_obs[0].layerArray[i].bias.data.clone()     
        
            l1_norms = torch.sum(torch.abs(original_weights), dim=0)   
        
            _, top_indices = torch.topk(l1_norms, 2*new_input_size)      
        
            self.nhidden = new_input_size
        
            new_input_layer = nn.Linear(2 * self.nhidden, 50)
        
            new_input_layer.weight.data = original_weights[:, top_indices] 
            new_input_layer.bias.data = original_biases  
            
            self.z0_to_obs[0].layerArray[i] = new_input_layer


    def prune_decoder_layers(self, threshold=0.05):
        # Prune first layer's output in z0_to_obs
        for i in range(len(self.z0_to_obs[0].layerArray)):
            original_weights_1 = self.z0_to_obs[0].layerArray[i].weight.data.clone()
            original_biases_1 = self.z0_to_obs[0].layerArray[i].bias.data.clone()
            l1_norms_1 = torch.sum(torch.abs(original_weights_1), dim=1)
            top_k_1 = int((1 - threshold) * l1_norms_1.size(0))
            _, top_indices_1 = torch.topk(l1_norms_1, top_k_1)
            
            new_layer_1 = nn.Linear(original_weights_1.size(1), top_k_1).to(self.device)
            new_layer_1.weight.data = original_weights_1[top_indices_1, :]
            new_layer_1.bias.data = original_biases_1[top_indices_1]
            self.z0_to_obs[0].layerArray[i] = new_layer_1

            # Prune second layer's input in z0_to_obs
            original_weights_2 = self.z0_to_obs[2].layerArray[i].weight.data.clone()
            original_biases_2 = self.z0_to_obs[2].layerArray[i].bias.data.clone()
            l1_norms_2 = torch.sum(torch.abs(original_weights_2), dim=0)
            top_k_2 = int((1 - threshold) * l1_norms_2.size(0))
            _, top_indices_2 = torch.topk(l1_norms_2, top_k_2)
            
            new_layer_2 = nn.Linear(top_k_2, original_weights_2.size(0)).to(self.device)
            new_layer_2.weight.data = original_weights_2[:, top_indices_2]
            new_layer_2.bias.data = original_biases_2
            self.z0_to_obs[2].layerArray[i] = new_layer_2

        # print(self.z0_to_obs)
        # self.prune_decoder_layers()

    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, z, time_steps):
        self.query = self.query.to(self.device)
        out, _ = self.gru_rnn(z)
        #time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.embedder1(time_steps).to(self.device)
            key = self.embedder2(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out        
   
       
   
    
# class enc_mtan_classif(nn.Module):
 
#     def __init__(self, input_dim, query, nhidden=16, 
#                  embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
#         super(enc_mtan_classif, self).__init__()
#         assert embed_time % num_heads == 0
#         self.freq = freq
#         self.embed_time = embed_time
#         self.learn_emb = learn_emb
#         self.dim = input_dim
#         self.device = device
#         self.nhidden = nhidden
#         self.query = query
#         self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
#         self.classifier = nn.Sequential(
#             nn.Linear(nhidden, 300),
#             nn.ReLU(),
#             nn.Linear(300, 300),
#             nn.ReLU(),
#             nn.Linear(300, 2))
#         self.enc = nn.GRU(nhidden, nhidden)
#         if learn_emb:
#             self.periodic = nn.Linear(1, embed_time-1)
#             self.linear = nn.Linear(1, 1)
    
    
#     def learn_time_embedding(self, tt):
#         tt = tt.to(self.device)
#         tt = tt.unsqueeze(-1)
#         out2 = torch.sin(self.periodic(tt))
#         out1 = self.linear(tt)
#         return torch.cat([out1, out2], -1)
       
        
#     def time_embedding(self, pos, d_model):
#         pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
#         position = 48.*pos.unsqueeze(2)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(np.log(self.freq) / d_model))
#         pe[:, :, 0::2] = torch.sin(position * div_term)
#         pe[:, :, 1::2] = torch.cos(position * div_term)
#         return pe
    
       
#     def forward(self, x, time_steps):
#         time_steps = time_steps.cpu()
#         mask = x[:, :, self.dim:]
#         mask = torch.cat((mask, mask), 2)
#         if self.learn_emb:
#             key = self.learn_time_embedding(time_steps).to(self.device)
#             query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
#         else:
#             key = self.time_embedding(time_steps, self.embed_time).to(self.device)
#             query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
            
#         out = self.att(query, key, x, mask)
#         out = out.permute(1, 0, 2)
#         _, out = self.enc(out)
#         return self.classifier(out.squeeze(0))




# class enc_mtan_classif_activity(nn.Module):
 
#     def __init__(self, input_dim, nhidden=16, 
#                  embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
#         super(enc_mtan_classif_activity, self).__init__()
#         assert embed_time % num_heads == 0
#         self.freq = freq
#         self.embed_time = embed_time
#         self.learn_emb = learn_emb
#         self.dim = input_dim
#         self.device = device
#         self.nhidden = nhidden
#         self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
#         self.gru = nn.GRU(nhidden, nhidden, batch_first=True)
#         self.classifier = nn.Linear(nhidden, 11)
#         if learn_emb:
#             self.periodic = nn.Linear(1, embed_time-1)
#             self.linear = nn.Linear(1, 1)
    
    
#     def learn_time_embedding(self, tt):
#         tt = tt.to(self.device)
#         tt = tt.unsqueeze(-1)
#         out2 = torch.sin(self.periodic(tt))
#         out1 = self.linear(tt)
#         return torch.cat([out1, out2], -1)
       
        
#     def time_embedding(self, pos, d_model):
#         pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
#         position = 48.*pos.unsqueeze(2)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(np.log(self.freq) / d_model))
#         pe[:, :, 0::2] = torch.sin(position * div_term)
#         pe[:, :, 1::2] = torch.cos(position * div_term)
#         return pe
    
       
#     def forward(self, x, time_steps):
#         batch = x.size(0)
#         time_steps = time_steps.cpu()
#         mask = x[:, :, self.dim:]
#         mask = torch.cat((mask, mask), 2)
#         if self.learn_emb:
#             key = self.learn_time_embedding(time_steps).to(self.device)
#         else:
#             key = self.time_embedding(time_steps, self.embed_time).to(self.device)
#         out = self.att(key, key, x, mask)
#         out, _ = self.gru(out)
#         out = self.classifier(out)
#         return out
    
    
    
# class enc_interp(nn.Module):
#     def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device='cuda'):
#         super(enc_interp, self).__init__()
#         self.dim = input_dim
#         self.device = device
#         self.nhidden = nhidden
#         self.query = query
#         self.cross = nn.Linear(2*input_dim, 2*input_dim)
#         self.bandwidth = nn.Linear(1, 2*input_dim, bias=False)
#         self.gru_rnn = nn.GRU(2*input_dim, nhidden, bidirectional=True, batch_first=True)
#         self.hiddens_to_z0 = nn.Sequential(
#             nn.Linear(2*nhidden, 50),
#             nn.ReLU(),
#             nn.Linear(50, latent_dim * 2))

    
#     def attention(self, query, key, value, mask=None, dropout=None):
#         "Compute 'Scaled Dot Product Attention'"
#         query, key = query.to(self.device), key.to(self.device)
#         batch, seq_len, dim = value.size()
#         scores = -(query.unsqueeze(-1) - key.unsqueeze(-2))**2
#         scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
#         bandwidth = torch.log(1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1).to(self.device))))
#         scores = scores * bandwidth
#         if mask is not None:
#             scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
#         p_attn = F.softmax(scores, dim = -2)
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         #return torch.sum(p_attn*value, -2), p_attn
#         return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
        
#     def forward(self, x, time_steps):
#         #time_steps = time_steps.cpu()
#         mask = x[:, :, self.dim:]
#         mask = torch.cat((mask, mask), 2)
#         out, _ = self.attention(self.query.unsqueeze(0), time_steps, x, mask)
#         out = self.cross(out)
#         out, _ = self.gru_rnn(out)
#         out = self.hiddens_to_z0(out)
#         return out
    
    
# class dec_interp(nn.Module):
 
#     def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device='cuda'):
#         super(dec_interp, self).__init__()
#         self.dim = input_dim
#         self.device = device
#         self.nhidden = nhidden
#         self.query = query
#         self.bandwidth = nn.Linear(1, 2*nhidden, bias=False)
#         self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
#         self.z0_to_obs = nn.Sequential(
#             nn.Linear(2*nhidden, 50),
#             nn.ReLU(),
#             nn.Linear(50, input_dim))
        
    
#     def attention(self, query, key, value, mask=None, dropout=None):
#         "Compute 'Scaled Dot Product Attention'"
#         query, key = query.to(self.device), key.to(self.device)
#         batch, seq_len, dim = value.size()
#         scores = -(query.unsqueeze(-1) - key.unsqueeze(-2))**2
#         scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
#         bandwidth = torch.log(1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1).to(self.device))))
#         scores = scores * bandwidth
#         if mask is not None:
#             scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
#         p_attn = F.softmax(scores, dim = -2)
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         #return torch.sum(p_attn*value, -2), p_attn
#         return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
        
#     def forward(self, z, time_steps):
#         out, _ = self.gru_rnn(z)
#         out, _ = self.attention(time_steps, self.query.unsqueeze(0), out)
#         out = self.z0_to_obs(out)
#         return out        

    
# class enc_rnn3(nn.Module):
#     def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
#                  embed_time=16, use_classif=False, learn_emb=False, device='cuda'):
#         super(enc_rnn3, self).__init__()
#         self.use_classif = use_classif 
#         self.embed_time = embed_time
#         self.dim = input_dim
#         self.device = device
#         self.nhidden = nhidden
#         self.query = query
#         self.learn_emb = learn_emb
#         self.cross = nn.Linear(2*input_dim, nhidden)
#         if use_classif:
#             self.gru_rnn = nn.GRU(nhidden, nhidden, batch_first=True)
#         else:
#             self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
#         self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time) for _ in range(2)])
#         if learn_emb:
#             self.periodic = nn.Linear(1, embed_time-1)
#             self.linear = nn.Linear(1, 1)
#         if use_classif:
#             self.classifier = nn.Sequential(
#                 nn.Linear(nhidden, 300),
#                 nn.ReLU(),
#                 nn.Linear(300, 300),
#                 nn.ReLU(),
#                 nn.Linear(300, 2))
#         else:
#             self.hiddens_to_z0 = nn.Sequential(
#                 nn.Linear(2*nhidden, 50),
#                 nn.ReLU(),
#                 nn.Linear(50, latent_dim * 2))
    
#     def learn_time_embedding(self, tt):
#         tt = tt.to(self.device)
#         tt = tt.unsqueeze(-1)
#         out2 = torch.sin(self.periodic(tt))
#         out1 = self.linear(tt)
#         return torch.cat([out1, out2], -1)
    
    
#     def fixed_time_embedding(self, pos):
#         d_model=self.embed_time
#         pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
#         position = 48.*pos.unsqueeze(2)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(np.log(10.0) / d_model))
#         pe[:, :, 0::2] = torch.sin(position * div_term)
#         pe[:, :, 1::2] = torch.cos(position * div_term)
#         return pe
    
#     def attention(self, query, key, value, mask=None, dropout=None):
#         "Compute 'Scaled Dot Product Attention'"
#         batch, seq_len, dim = value.size()
#         d_k = query.size(-1)
#         query, key = [l(x) for l, x in zip(self.linears, (query, key))]
#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#                  / math.sqrt(d_k)
#         scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
#         if mask is not None:
#             scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
#         p_attn = F.softmax(scores, dim = -2)
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         #return torch.sum(p_attn*value, -2), p_attn
#         return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
#     def forward(self, x, time_steps):
#         time_steps = time_steps.cpu()
#         mask = x[:, :, self.dim:]
#         mask = torch.cat((mask, mask), 2)
#         if self.learn_emb:
#             key = self.learn_time_embedding(time_steps).to(self.device)
#             query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
#         else:
#             key = self.fixed_time_embedding(time_steps).to(self.device)
#             query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device) 
#         out, _ = self.attention(query, key, x, mask)
#         out = self.cross(out)
#         if not self.use_classif:
#             out, _ = self.gru_rnn(out)
#             out = self.hiddens_to_z0(out)
#         else:
#             _, h = self.gru_rnn(out)
#             out = self.classifier(h.squeeze(0))
#         return out
    
    
# class dec_rnn3(nn.Module):
 
#     def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
#                  embed_time=16,learn_emb=False, device='cuda'):
#         super(dec_rnn3, self).__init__()
#         self.embed_time = embed_time
#         self.dim = input_dim
#         self.device = device
#         self.nhidden = nhidden
#         self.query = query
#         self.learn_emb = learn_emb
#         self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
#         self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
#                                      nn.Linear(embed_time, embed_time),
#                                      nn.Linear(2*nhidden, 2*nhidden)])
      
#         if learn_emb:
#             self.periodic = nn.Linear(1, embed_time-1)
#             self.linear = nn.Linear(1, 1)
#         self.z0_to_obs = nn.Sequential(
#             nn.Linear(2*nhidden, 50),
#             nn.ReLU(),
#             nn.Linear(50, input_dim))
        
#     def learn_time_embedding(self, tt):
#         tt = tt.to(self.device)
#         tt = tt.unsqueeze(-1)
#         out2 = torch.sin(self.periodic(tt))
#         out1 = self.linear(tt)
#         return torch.cat([out1, out2], -1)
        
        
#     def fixed_time_embedding(self, pos):
#         d_model = self.embed_time
#         pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
#         position = 48.*pos.unsqueeze(2)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(np.log(10.0) / d_model))
#         pe[:, :, 0::2] = torch.sin(position * div_term)
#         pe[:, :, 1::2] = torch.cos(position * div_term)
#         return pe
    
#     def attention(self, query, key, value, mask=None, dropout=None):
#         "Compute 'Scaled Dot Product Attention'"
#         batch, seq_len, dim = value.size()
#         d_k = query.size(-1)
#         query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#                  / math.sqrt(d_k)
#         scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
#         if mask is not None:
#             scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
#         p_attn = F.softmax(scores, dim = -2)
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         #return torch.sum(p_attn*value, -2), p_attn
#         return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
#     def forward(self, z, time_steps):
#         out, _ = self.gru_rnn(z)
#         time_steps = time_steps.cpu()
#         if self.learn_emb:
#             query = self.learn_time_embedding(time_steps).to(self.device)
#             key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
#         else:
#             query = self.fixed_time_embedding(time_steps).to(self.device)
#             key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
#         out, _ = self.attention(query, key, out)
#         out = self.z0_to_obs(out)
#         return out        
    