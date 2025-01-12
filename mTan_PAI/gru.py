import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size) * 0.1)
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size) * 0.1)
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))

    def update_hidden_size(self, new_hidden_size):
        self.hidden_size = new_hidden_size

    def forward(self, x, h_prev):
        # Ensure x has shape (batch_size, input_size)
        x = x.view(-1, x.size(-1))  # Flatten if needed
        
        # Ensure h_prev has shape (batch_size, hidden_size)
        h_prev = h_prev.view(x.size(0), -1)

        # Compute gates for the entire batch at once
        gate_x = torch.matmul(x, self.weight_ih.t()) + self.bias_ih  # (batch_size, 3 * hidden_size)
        gate_h = torch.matmul(h_prev, self.weight_hh.t()) + self.bias_hh  # (batch_size, 3 * hidden_size)

        # Split gates for update (z), reset (r), and new hidden state (h_tilde)
        z_x, r_x, h_tilde_x = torch.split(gate_x, self.hidden_size, dim=-1)
        z_h, r_h, h_tilde_h = torch.split(gate_h, self.hidden_size, dim=-1)

        # Compute gates
        z = torch.sigmoid(z_x + z_h)  # (batch_size, hidden_size)
        r = torch.sigmoid(r_x + r_h)  # (batch_size, hidden_size)
        h_tilde = torch.tanh(h_tilde_x + r * h_tilde_h)  # (batch_size, hidden_size)

        # Compute the new hidden state
        h_t = (1 - z) * h_prev + z * h_tilde  # (batch_size, hidden_size)
        return h_t


class StandardGRU(nn.Module):
    def __init__(self, input_size=20, hidden_size=12):
        super(StandardGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = GRUCell(input_size, hidden_size)

    def forward(self, X):
        batch_size, seq_length, _ = X.shape
        hidden_size = self.gru_cell.hidden_size

        h_all = torch.zeros(batch_size, seq_length, hidden_size).to(X.device)
        final_h = torch.zeros(batch_size, hidden_size).to(X.device)
        
        h_prev = torch.zeros(batch_size, hidden_size, device=X.device)
      
        for t in range(seq_length):
            h_prev = self.gru_cell(X[:, t], h_prev)  
            h_all[:, t] = h_prev 

        # The final hidden state is the last time step's hidden state for each sequence
        final_h = h_prev

        return h_all, final_h

class BidirectionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BidirectionalGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forward_gru = GRUCell(input_size, hidden_size)
        self.backward_gru = GRUCell(input_size, hidden_size)

  

    def forward(self, X):
        batch_size, seq_length, _ = X.shape
        hidden_size = self.forward_gru.hidden_size

        h_forward = torch.zeros(batch_size, seq_length, hidden_size).to(X.device)
        h_backward = torch.zeros(batch_size, seq_length, hidden_size).to(X.device)

        final_h_forward = torch.zeros(batch_size, hidden_size).to(X.device)
        final_h_backward = torch.zeros(batch_size, hidden_size).to(X.device)

        h_prev_forward = torch.zeros(batch_size, hidden_size).to(X.device)
        h_prev_backward = torch.zeros(batch_size, hidden_size).to(X.device)
      
        for t in range(seq_length):
            h_prev_forward = self.forward_gru(X[:, t], h_prev_forward)
            h_forward[:, t] = h_prev_forward

        for t in reversed(range(seq_length)):
            h_prev_backward = self.backward_gru(X[:, t], h_prev_backward)
            h_backward[:, t] = h_prev_backward

        H = torch.cat((h_forward, h_backward), dim=2)
        final_hidden = torch.cat((final_h_forward, final_h_backward), dim=1) 

        return H, final_hidden  # Return both outputs
    
def prune_standard_gru(gru_component):

    layers = []
    for name, layer in gru_component.named_parameters():
        layers.append(str(name))
    

    ih_l0 = gru_component.gru_cell.weight_ih.data
    ih_l0_bias = gru_component.gru_cell.bias_ih.data

    hh_l0 = gru_component.gru_cell.weight_hh.data
    hh_l0_bias = gru_component.gru_cell.bias_hh.data

    index = ih_l0.shape[0]//3

    W_ir = ih_l0[:index, :]  # Reset gate weights
    W_iz = ih_l0[index:index*2, :]  # Update gate weights
    W_in = ih_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

    threshold = 0.96
    num_units_to_keep = int(threshold * index)  # 80% of 256
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.gru_cell.weight_ih.data = pruned_weight_ih

    # ih bias
    W_ir = ih_l0_bias[:index]  # Reset gate weights
    W_iz = ih_l0_bias[index:index*2]  # Update gate weights
    W_in = ih_l0_bias[index*2:]  # New gate weights

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.gru_cell.bias_ih.data = pruned_weight_ih

    # hh
    W_ir = hh_l0[:index, :]  # Reset gate weights
    W_iz = hh_l0[index:index*2, :]  # Update gate weights
    W_in = hh_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

    num_units_to_keep = int(threshold * index)
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.gru_cell.weight_hh.data = pruned_weight_ih

    # hh bias
    W_ir = hh_l0_bias[:index]  # Reset gate weights
    W_iz = hh_l0_bias[index:index*2]  # Update gate weights
    W_in = hh_l0_bias[index*2:]  # New gate weights

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.gru_cell.bias_hh.data = pruned_weight_ih

    
    
    return gru_component

def prune_bi_gru(gru_component):

    layers = []
    for name, layer in gru_component.named_parameters():
        layers.append(str(name))
    
    

    ih_l0 = gru_component.forward_gru.weight_ih.data
    ih_l0_bias = gru_component.forward_gru.bias_ih.data

    hh_l0 = gru_component.forward_gru.weight_hh.data
    hh_l0_bias = gru_component.forward_gru.bias_hh.data

    if "backward_gru.weight_ih" in layers:
        ih_l0_reverse = gru_component.backward_gru.weight_ih.data
        ih_l0_reverse_bias = gru_component.backward_gru.bias_ih.data

        hh_l0_reverse = gru_component.backward_gru.weight_hh.data
        hh_l0_reverse_bias = gru_component.backward_gru.bias_hh.data

    index = ih_l0.shape[0]//3

  
    W_ir = ih_l0[:index, :]  # Reset gate weights
    W_iz = ih_l0[index:index*2, :]  # Update gate weights
    W_in = ih_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

    threshold = 0.96
    num_units_to_keep = int(threshold * index)  # 80% of 256
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.forward_gru.weight_ih.data = pruned_weight_ih

    # ih bias
    W_ir = ih_l0_bias[:index]  # Reset gate weights
    W_iz = ih_l0_bias[index:index*2]  # Update gate weights
    W_in = ih_l0_bias[index*2:]  # New gate weights

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.forward_gru.bias_ih.data = pruned_weight_ih

    # hh
    W_ir = hh_l0[:index, :]  # Reset gate weights
    W_iz = hh_l0[index:index*2, :]  # Update gate weights
    W_in = hh_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

    num_units_to_keep = int(threshold * index)
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.forward_gru.weight_hh.data = pruned_weight_ih

    # hh bias
    W_ir = hh_l0_bias[:index]  # Reset gate weights
    W_iz = hh_l0_bias[index:index*2]  # Update gate weights
    W_in = hh_l0_bias[index*2:]  # New gate weights

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.forward_gru.bias_hh.data = pruned_weight_ih

    new_index = ih_l0_reverse.shape[0]//3

    gru_component.hidden_size = new_index

    if "backward_gru.weight_ih" in layers:

        W_ir = ih_l0_reverse[:index, :]  # Reset gate weights
        W_iz = ih_l0_reverse[index:index*2, :]  # Update gate weights
        W_in = ih_l0_reverse[index*2:, :]  # New gate weights

        l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
        l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
        l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

        threshold = 0.96
        num_units_to_keep = int(threshold * index)  # 80% of 256
        keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
        keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
        keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[keep_indices_ir]
        W_iz_pruned = W_iz[keep_indices_iz]
        W_in_pruned = W_in[keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.backward_gru.weight_ih.data = pruned_weight_ih

        W_ir = ih_l0_reverse_bias[:index]  # Reset gate weights
        W_iz = ih_l0_reverse_bias[index:index*2]  # Update gate weights
        W_in = ih_l0_reverse_bias[index*2:]  # New gate weights

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[keep_indices_ir]
        W_iz_pruned = W_iz[keep_indices_iz]
        W_in_pruned = W_in[keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.backward_gru.bias_ih.data = pruned_weight_ih

        W_ir = hh_l0_reverse[:index, :]  # Reset gate weights
        W_iz = hh_l0_reverse[index:index*2, :]  # Update gate weights
        W_in = hh_l0_reverse[index*2:, :]  # New gate weights

        l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
        l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
        l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

        threshold = 0.96
        num_units_to_keep = int(threshold * index)  # 80% of 256
        keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
        keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
        keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[keep_indices_ir]
        W_iz_pruned = W_iz[keep_indices_iz]
        W_in_pruned = W_in[keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.backward_gru.weight_hh.data = pruned_weight_ih

        W_ir = hh_l0_reverse_bias[:index]  # Reset gate weights
        W_iz = hh_l0_reverse_bias[index:index*2]  # Update gate weights
        W_in = hh_l0_reverse_bias[index*2:]  # New gate weights

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[keep_indices_ir]
        W_iz_pruned = W_iz[keep_indices_iz]
        W_in_pruned = W_in[keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.backward_gru.bias_hh.data = pruned_weight_ih

        new_index = ih_l0_reverse.shape[0]//3

    for name, param in gru_component.named_parameters():
        print(f"{name} : {param.shape}")

    
    
    return gru_component

def prune_hh_gru(gru_component):

    layers = []
    for name, layer in gru_component.named_parameters():
        layers.append(str(name))

    hh_l0 = gru_component.forward_gru.weight_hh.data

    if "backward_gru.weight_ih" in layers:
        hh_l0_reverse = gru_component.backward_gru.weight_hh.data

    index = hh_l0.shape[1]
    # print(f"Index: ",index)

    threshold = 0.96
    num_units_to_keep = int(threshold * index) 
    
    # hh
    W_ir = hh_l0[:index, :]  # Reset gate weights
    W_iz = hh_l0[index:index*2, :]  # Update gate weights
    W_in = hh_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=0)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=0)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=0)  # L1 norm for new gate

   
    num_units_to_keep = int(threshold * index)
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[:,keep_indices_ir]
    W_iz_pruned = W_iz[:,keep_indices_iz]
    W_in_pruned = W_in[:,keep_indices_in]

    pruned_weight_hh = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.forward_gru.weight_hh.data = pruned_weight_hh

    if "backward_gru.weight_ih" in layers:

        W_ir = hh_l0_reverse[:index, :]  # Reset gate weights
        W_iz = hh_l0_reverse[index:index*2, :]  # Update gate weights
        W_in = hh_l0_reverse[index*2:, :]  # New gate weights

        l1_norm_ir = W_ir.abs().sum(dim=0)  # L1 norm for reset gate
        l1_norm_iz = W_iz.abs().sum(dim=0)  # L1 norm for update gate
        l1_norm_in = W_in.abs().sum(dim=0)  # L1 norm for new gate

        threshold = 0.96
        num_units_to_keep = int(threshold * index)  
        keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
        keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
        keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[:,keep_indices_ir]
        W_iz_pruned = W_iz[:,keep_indices_iz]
        W_in_pruned = W_in[:,keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.backward_gru.weight_hh.data = pruned_weight_ih
    
    return gru_component

def prune_hh_standard_gru(gru_component):

    hh_l0 = gru_component.gru_cell.weight_hh.data

    index = hh_l0.shape[1]
    # print(f"Index: ",index)

    threshold = 0.96
    num_units_to_keep = int(threshold * index) 
    
    # hh
    W_ir = hh_l0[:index, :]  # Reset gate weights
    W_iz = hh_l0[index:index*2, :]  # Update gate weights
    W_in = hh_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=0)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=0)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=0)  # L1 norm for new gate

    num_units_to_keep = int(threshold * index)
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[:,keep_indices_ir]
    W_iz_pruned = W_iz[:,keep_indices_iz]
    W_in_pruned = W_in[:,keep_indices_in]

    pruned_weight_hh = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.gru_cell.weight_hh.data = pruned_weight_hh

 
    return gru_component

if __name__ == "__main__":

    input_size = 256
    hidden_size = 128
    seq_length = 278
    batch_size = 50

    bi_gru = BidirectionalGRU(input_size, hidden_size)

    X = torch.randn(batch_size, seq_length, input_size)

    print("Named Parameters of Bidirectional GRU:")
    for name, param in bi_gru.named_parameters():
        print(f"{name}: {param.shape}")

    H = bi_gru(X)
    print("\nConcatenated Hidden States (Forward + Backward):\n", H.shape)



