import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj 

def get_gnn_layer(gnn_type, embed_dim=64):
    if gnn_type == 'gcn':
        return torch_geometric.nn.GCNConv(embed_dim, embed_dim)
    elif gnn_type =='gin':
        mlp = mlp = nn.Sequential( nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        return torch_geometric.nn.GINConv(mlp, train_eps=True)
    else:
        raise ValueError("Not implemented!")

def mvm(A, b):
    """ Multiplies each row of b by A
    """
    return A.matmul(b.unsqueeze(-1)).squeeze()

def matrix_cosine(a, b, eps=1e-8):
    """
    Pairwise cosine of embeddings
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class MICRO_Graph(torch.nn.Module):
    def __init__(self, encoder, n_motifs=20, n_hid=3, tau=1, alpha=1) :
        super(MICRO_Graph , self).__init__ ()
        self.enc = encoder # Any GNN Model could be used
        self.M = torch.nn.Parameter(torch.rand((n_motifs, n_hid))) # Motif Embedding Table
        self.W_h = torch.nn.Parameter(torch.rand((n_hid, n_hid)))
        self.W_s = torch.nn.Parameter(torch.rand((n_hid, n_hid)))
        self.W_e = torch.nn.Parameter(torch.rand((n_hid, n_hid)))
        self.tau = tau
        self.alpha = alpha

    def forward(self, data):
        h, e = self.enc(data) # node and graph embeddings
        # Get Motif - like subgraphs via Partition
        # Detach self .M to stop grad flow for Motif here
        Q = matrix_cosine(mvm(self.W_h, h), self.M.detach())
        Q = torch.softmax(Q/self.tau , dim = -1) # N_B x K
        with torch.no_grad():
            # Don 't store gradient for discrete ops Q_hat = sinkhorn(Q)
            Q_hat = self.sinkhorn(Q)
            # num_subs : number of subgraphs sampled from each G_i .
            s, P_hat, num_subs = self.pool_sub(h, Q_hat)

        # Calculate motif -to - subgraph score . Detach GNN to avoid degeneration solution .
        # equation 12
        P = matrix_cosine(mvm(self.W_s, s.detach()), self.M )
        P = torch.softmax( P / self.tau , dim = -1) # J_B x K

        # Calculate the two loss via M- step .
        loss_m = self.motif_loss(Q , Q_hat, P, P_hat , to_dense_adj(data.edge_index, data.batch))
        loss_c = self.contra_loss(s , e , num_subs)
        return self.alpha * loss_m + (1 - self.alpha ) * loss_c

    def pool_sub(self, h, Q_hat, min_motif_size=-1):
        """ Compute embeddings for each subgraph
        Q_hat maps each node to one of the K motifs
        For all nodes mapped to the same motif, pool the embeddings.
        """
        node_to_mot = torch.nonzero(Q_hat.t())
        s = torch.zeros(Q_hat.shape[0], h.shape[1])
        for motif in range(Q_hat.shape[0]):
            nodes = Q_hat[motif].nonzero().squeeze()
            s[motif] = h[nodes].mean(dim=0).squeeze()

        # filter out sparsely populated motifs
        keep = (Q_hat.sum(dim=1) > min_motif_size).nonzero().t().squeeze()
        s = s[keep]
        P_hat = Q_hat[keep]
        print(P_hat.shape)
        print(len(keep))

        return s, P_hat.t(), len(keep)

    def motif_loss(self, Q, Q_hat, P, P_hat, adj):
        loss_mot_sub = -(P_hat * P.log()).sum(dim =1).mean() # Eq (13)
        loss_node_mot = -(Q_hat * Q.log()).sum(dim=1).mean() # Eq (10)
        loss_reg = spectral_loss(Q, adj) # Eq (11)
        return loss_mot_sub + loss_node_mot + loss_reg

    def contra_loss(self, s, e, num_subs):
        blocks = [torch.ones(1 , n ) for n in num_subs]
        Y_lab = torch.block_diag(*blocks)
        Y = pairwise_cosine_sim(self.W_e * e , s)
        Y = torch.softmax( Y / self.tau, dim=-1)
        loss_contra = - ( Y_lab * Y.log()).sum(dim =1).mean() # Eq (15)
        return loss_contra

    def sinkhorn(self, Q , num_iters=5, lamb =20):
        # Implementation adopted from https :// github . com / facebookresearch / swav
        Q = Q.transpose(0 ,1) # K x N_B
        Q = torch.exp(Q*lamb)
        Q /= torch.sum(Q)
        u = torch.zeros(Q.shape[0])
        r = torch.ones(Q.shape[0]) / Q.shape[0]
        c = torch.ones(Q.shape[1]) / Q.shape[1]
        curr_sum = torch.sum(Q, dim =1)

        for it in range(num_iters):
            u = curr_sum
            Q *= (r/u).unsqueeze (1)
            Q *= (c/torch.sum(Q , dim =0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim =1)

        Q_hat = (Q / torch.sum(Q , dim =0 , keepdim=True))
        Q_hat = F.one_hot(Q_hat.argmax(dim =0))
        return Q_hat.transpose(0 ,1) # N_B x K

class GNN(torch.nn.Module):
    def __init__(self, in_dim=1, num_class=1, embed_dim=64, gnn_type='gcn', num_layers=2, dropout=0.0, global_pool='sum'):
        super().__init__()
        self.num_class = num_class
        self.gnn_type = gnn_type
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.global_pool = global_pool

        self.in_dim = in_dim
        if in_dim == 1:
            self.node_embed = torch.nn.Embedding(in_dim, embed_dim)
        else:
            self.node_embed = torch.nn.Linear(in_dim, embed_dim, bias=False)

        self.conv_layers = []
        self.bn_layers = []

        for i in range(num_layers):
            conv_layer = get_gnn_layer(gnn_type, embed_dim)
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(torch.nn.BatchNorm1d(self.embed_dim))


        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.bn_layers = torch.nn.ModuleList(self.bn_layers)

        self.dropout = dropout

        if global_pool == 'sum':
            self.global_pool = torch_geometric.nn.global_add_pool
        elif global_pool == 'mean':
            self.global_pool = torch_geometric.nn.global_mean_pool
        elif global_pool == 'max':
            self.global_pool = torch_geometric.nn.global_max_pool

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.node_embed(x)

        for i in range(self.num_layers):
            h = self.conv_layers[i](h, edge_index)
            h = self.bn_layers[i](h)

            if i < self.num_layers - 1:
                h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)

        h_pool = self.global_pool(h, data.batch)
        return h, h_pool


if __name__ == "__main__":
    epochs = 10
    batch_size = 8
    embed_dim = 32
    
    data = TUDataset(root='.', name='PROTEINS')
    loader = DataLoader(data, batch_size=batch_size)

    enc = GNN(in_dim=data.num_features, embed_dim=embed_dim) # Any GNN Model could be used
    model = MICRO_Graph(enc, n_hid=embed_dim)

    for epoch in range(epochs):
        for batch in loader:
            out = model(batch)
            print(out)
