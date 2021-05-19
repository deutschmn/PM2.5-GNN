import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear
import numpy as np
from torch.nn import functional as F

class SplitGNN_5(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std, node_gru_hidden_dim, edge_gru_hidden_dim, edge_mlp_hidden_dim):
        super(SplitGNN_5, self).__init__()

        self.returns_r = True

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len

        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        
        self.num_nodes = city_num
        self.num_edges = self.edge_index.size(1)

        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = node_gru_hidden_dim

        self.gru_cell = GRUCell(self.in_dim + 1, self.hid_dim) # in features + PM2.5 from prev. iteration
        self.node_mlp = Linear(self.hid_dim, 1)
        
        # graph attributes
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)

        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)

        self.edge_mlp_hidden_dim = edge_mlp_hidden_dim
        self.edge_mlp = Sequential(Linear(3, self.edge_mlp_hidden_dim),
                                   nn.ReLU(),
                                   Linear(self.edge_mlp_hidden_dim, 1))

        # used in restructure_edges to restructure from edge to matrix shape
        self.edge_restructure_idx = self.edge_index[0] + (self.edge_index[1] * self.num_nodes)
        self.edge_restructure_idx = self.edge_restructure_idx.repeat(self.batch_size).view(self.batch_size, -1)

        self.R_0 = torch.load("data/R_corr_DS2.pt")
        self.R_0 = self.R_0.repeat_interleave(self.batch_size).view(self.num_nodes, self.num_nodes, self.batch_size).permute(2, 0, 1)
        self.R_0 = self.R_0.to(self.device).float()
        self.a = nn.parameter.Parameter(torch.tensor([0.0])) # controls trade-off between R_0 and R_w


    def generate_edge_attributes(self, wind):
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)

        edge_src, edge_target = self.edge_index
        node_src = wind[:, edge_src]
        node_target = wind[:, edge_target]

        src_wind = node_src * self.wind_std[None,None,:] + self.wind_mean[None,None,:]
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:,:,1]
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(self.batch_size, 1, 1)
        city_dist = self.edge_attr_[:,:,0]
        city_direc = self.edge_attr_[:,:,1]

        theta = torch.abs(city_direc - src_wind_direc)
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist) # advection S [eq. 4]
        edge_weight = edge_weight.to(self.device)
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(self.batch_size, 1, 1).to(self.device)
        edge_atts = torch.cat([edge_attr_norm, edge_weight[:,:,None]], dim=-1)

        return edge_atts


    def restructure_edges(self, e_rep):
        """ Generates R matrix from edge representations """
        # Old implementation that's much slower, but probably easier to understand:
        # R = torch.zeros(self.batch_size, self.num_nodes, self.num_nodes, device=self.device)
        # for e in range(self.num_edges):
        #     src, sink = self.edge_index[:, e]
        #     R[:, sink.item(), src.item()] = e_rep[:, e].squeeze()
        R = torch.zeros(self.batch_size, self.num_nodes * self.num_nodes, device=self.device)
        R = R.scatter_(1, self.edge_restructure_idx, e_rep).view(self.batch_size, self.num_nodes, self.num_nodes)
        return R

    @property
    def alpha(self):
        return F.sigmoid(self.a)

    def forward(self, pm25_hist, feature):
        assert feature.shape[:3] == (self.batch_size, self.hist_len + self.pred_len, self.num_nodes)
        assert pm25_hist.shape == (self.batch_size, self.hist_len, self.num_nodes, 1)

        pm25_pred = []
        R_list = []

        h0 = torch.zeros(self.batch_size * self.num_nodes, self.hid_dim).to(self.device)
        hn = h0

        c0 = torch.zeros(self.batch_size, self.num_nodes, 1).to(self.device)
        cn = c0

        for i in range(self.pred_len):
            x = feature[:, self.hist_len + i]
            if self.hist_len > 0:
                x = torch.cat((pm25_hist[:, -1], x), dim=-1)
            x = x.contiguous()

            # compute transfers
            current_wind = x[:,:, -2:]
            edge_atts = self.generate_edge_attributes(current_wind)
            en_rep = self.edge_mlp(edge_atts).squeeze()
            R_w = self.restructure_edges(en_rep)
            R_w = torch.softmax(R_w, dim=2)

            R = self.alpha * self.R_0 + (1 - self.alpha) * R_w
            R = torch.softmax(R, dim=2)
            R_list.append(R)

            # compute local phenomena
            node_in = torch.cat([x, cn], dim=2)
            hn = self.gru_cell(node_in, hn)
            hn_reshaped = hn.view(self.batch_size, self.num_nodes, self.hid_dim)
            hn_reshaped = self.node_mlp(hn_reshaped)
            
            # execute transfers
            cn = torch.matmul(R, hn_reshaped)

            pm25_pred.append(cn)

        R_list = torch.stack(R_list, dim=1)
        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred, R_list
