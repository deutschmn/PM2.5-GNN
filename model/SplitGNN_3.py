import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter


class GraphGNN(nn.Module):
    def __init__(self, device, edge_index, edge_attr, wind_mean, wind_std, batch_size):
        super(GraphGNN, self).__init__()
        self.device = device
        self.edge_index = torch.LongTensor(edge_index).to(self.device)

        self.num_nodes = self.edge_index.max().item() + 1
        self.num_edges = self.edge_index.size(1)

        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)

        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)

        self.batch_size = batch_size

        self.edge_mlp_hidden_dim = 32
        self.edge_gru_hidden_dim = 16
        self.edge_gru = GRUCell(3, self.edge_gru_hidden_dim)
        self.edge_mlp = Sequential(Linear(self.edge_gru_hidden_dim, self.edge_mlp_hidden_dim),
                                   nn.ReLU(),
                                   Linear(self.edge_mlp_hidden_dim, 1),
                                   nn.ReLU())

        e0 = torch.zeros(self.batch_size * self.num_edges, self.edge_gru_hidden_dim).to(self.device)
        self.e = e0

    def forward(self, x):
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)

        edge_src, edge_target = self.edge_index
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]

        # TODO make sure we're really using the wind here and not some other feature
        # FIXME really, we're adding wind_std to the wind direction?
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

        # FIXME this doesn't work -> have to detach self.e in the next iteration from comp. graph
        self.e = self.edge_gru(edge_atts, self.e)
        e_in = self.e.view(self.batch_size, self.num_edges, self.edge_gru_hidden_dim)
        e_rep = self.edge_mlp(e_in).squeeze()

        R = torch.zeros(self.batch_size, self.num_nodes, self.num_nodes, device=self.device)
        # TODO optimize this loop
        for e in range(self.num_edges):
            src, sink = self.edge_index[:, e]
            R[:, src.item(), sink.item()] = e_rep[:, e].squeeze()
        R = torch.softmax(R, dim=2)

        return R


class SplitGNN_3(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(SplitGNN_3, self).__init__()

        self.returns_r = True

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 1

        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, wind_mean, wind_std, batch_size)
        self.gru_cell = GRUCell(self.in_dim, self.hid_dim)
        # self.fc_out = nn.Linear(self.hid_dim, 1)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        R_list = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0

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

            R = self.graph_gnn(x)
            R_list.append(R)

            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            # xn = self.fc_out(xn)
            c = torch.matmul(R, xn)
            pm25_pred.append(c)

        R_list = torch.stack(R_list, dim=1)
        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred, R_list
