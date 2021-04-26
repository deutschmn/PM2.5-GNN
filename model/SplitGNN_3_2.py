import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter

class SplitGNN_3_2(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(SplitGNN_3_2, self).__init__()

        self.returns_r = True

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len

        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        
        self.num_nodes = city_num
        self.num_edges = self.edge_index.size(1)

        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 32

        self.in_indices = [0, 1, 2, 3, 4, 7, 8] # indices that we take from x (= all except wind)
        self.gru_cell = GRUCell(len(self.in_indices) + 1, self.hid_dim)
        self.node_mlp = Linear(self.hid_dim, 1)
        
        # graph attributes
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)

        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)

        self.edge_mlp_hidden_dim = 128
        self.edge_gru_hidden_dim = 64
        self.edge_gru = GRUCell(3, self.edge_gru_hidden_dim)
        self.edge_mlp = Sequential(Linear(self.edge_gru_hidden_dim, self.edge_mlp_hidden_dim),
                                   nn.ReLU(),
                                   Linear(self.edge_mlp_hidden_dim, 1))

        # used in restructure_edges to restructure from edge to matrix shape
        self.edge_restructure_idx = self.edge_index[0] + (self.edge_index[1] * self.num_nodes)
        self.edge_restructure_idx = self.edge_restructure_idx.repeat(self.batch_size).view(self.batch_size, -1)


    def generate_edge_attributes(self, x):
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)

        edge_src, edge_target = self.edge_index
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]

        src_wind = node_src[:,:,-2:] * self.wind_std[None,None,:] + self.wind_mean[None,None,:]
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:,:,1]
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)
        city_dist = self.edge_attr_[:,:,0]
        city_direc = self.edge_attr_[:,:,1]

        theta = torch.abs(city_direc - src_wind_direc)
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist) # advection S [eq. 4]
        edge_weight = edge_weight.to(self.device)
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(x.size(0), 1, 1).to(self.device)
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


    def forward(self, pm25_hist, feature):
        pm25_pred = []
        R_list = []

        h0 = torch.zeros(self.batch_size * self.num_nodes, self.hid_dim).to(self.device)
        hn = h0

        e0 = torch.zeros(self.batch_size * self.num_edges, self.edge_gru_hidden_dim).to(self.device)
        en = e0

        c0 = torch.zeros(self.batch_size, self.num_nodes, 1).to(self.device)
        cn = c0

        if pm25_hist.shape[1] == 0: # not using PM2.5 at all
            xn = None
        else:
            xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            if pm25_hist.shape[1] == 0: # not using PM2.5 at all
                x = feature[:, self.hist_len + i]
            else:
                x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            x = x.contiguous()

            # compute transfers
            edge_atts = self.generate_edge_attributes(x)
            en = self.edge_gru(edge_atts, en)
            en_reshaped = en.view(self.batch_size, self.num_edges, self.edge_gru_hidden_dim)
            en_rep = self.edge_mlp(en_reshaped).squeeze()
            R = self.restructure_edges(en_rep)
            R = torch.softmax(R, dim=2)
            R_list.append(R)

            # compute local phenomena 
            node_in = torch.cat([x[:,:,self.in_indices], cn], dim=2)
            hn = self.gru_cell(node_in, hn)
            hn_reshaped = hn.view(self.batch_size, self.num_nodes, self.hid_dim)
            hn_reshaped = self.node_mlp(hn_reshaped)
            
            # execute transfers
            cn = torch.matmul(R, hn_reshaped)

            pm25_pred.append(cn)

        R_list = torch.stack(R_list, dim=1)
        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred, R_list