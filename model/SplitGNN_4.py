import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear
import numpy as np
from torch.nn import functional as F

class TransferModel(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std, edge_gru_hidden_dim, edge_mlp_hidden_dim, node_module):
        super(TransferModel, self).__init__()

        assert hist_len == 1, "TransferModel only works with history of 1"

        self.returns_r = True

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim

        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        
        self.num_nodes = city_num
        self.num_edges = self.edge_index.size(1)

        self.batch_size = batch_size

        # graph attributes
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)

        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)

        self.edge_mlp_hidden_dim = edge_mlp_hidden_dim
        self.edge_gru_hidden_dim = edge_gru_hidden_dim
        self.edge_gru = GRUCell(3, self.edge_gru_hidden_dim)
        self.edge_mlp = Sequential(Linear(self.edge_gru_hidden_dim, self.edge_mlp_hidden_dim),
                                   nn.ReLU(),
                                   Linear(self.edge_mlp_hidden_dim, 1))

        # used in restructure_edges to restructure from edge to matrix shape
        self.edge_restructure_idx = self.edge_index[0] + (self.edge_index[1] * self.num_nodes)
        self.edge_restructure_idx = self.edge_restructure_idx.repeat(self.batch_size).view(self.batch_size, -1)

        self.node_module = node_module


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


    def forward(self, pm25, feature):
        wind = feature[:,:,:, -2:]

        assert wind.shape == (self.batch_size, self.hist_len + self.pred_len, self.num_nodes, 2)
        assert pm25.shape == (self.batch_size, self.node_module.expected_cn_len(self), self.num_nodes, 1)

        pm25_pred = []
        R_list = []

        e0 = torch.zeros(self.batch_size * self.num_edges, self.edge_gru_hidden_dim).to(self.device)
        en = e0

        self.node_module.reset_hn()

        for i in range(self.pred_len):
            # compute transfers
            current_wind = wind[:, self.hist_len + i]
            edge_atts = self.generate_edge_attributes(current_wind)
            en = self.edge_gru(edge_atts, en)
            en_reshaped = en.view(self.batch_size, self.num_edges, self.edge_gru_hidden_dim)
            en_rep = self.edge_mlp(en_reshaped).squeeze()
            R = self.restructure_edges(en_rep)
            R = torch.softmax(R, dim=2)
            R_list.append(R)
            
            # compute node phenomena (Oracle in pre-traning or Local in real training)
            cn = pm25
            hn_reshaped = self.node_module(cn, i, feature)

            # execute transfers
            c_pred = torch.matmul(R, hn_reshaped)

            pm25_pred.append(c_pred)

        R_list = torch.stack(R_list, dim=1)
        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred, R_list


class OracleModel(nn.Module):
    def __init__(self, node_gru_hidden_dim, city_num, batch_size, device):
        super(OracleModel, self).__init__()

        self.num_nodes = city_num
        self.batch_size = batch_size
        self.device = device
        self.hid_dim = node_gru_hidden_dim

        self.gru_cell = GRUCell(1, self.hid_dim)
        self.node_mlp = Linear(self.hid_dim, 1)

        self.reset_hn()

    def expected_cn_len(self, transfer_model):
        return transfer_model.hist_len + transfer_model.pred_len

    def reset_hn(self):
        h0 = torch.zeros(self.batch_size * self.num_nodes, self.hid_dim).to(self.device)
        self.hn = h0

    def forward(self, cn, i, _):
        node_in = cn[:, i - 1]
        node_in = node_in.contiguous()
        self.hn = self.gru_cell(node_in, self.hn)
        hn_reshaped = self.hn.view(self.batch_size, self.num_nodes, self.hid_dim)
        return self.node_mlp(hn_reshaped)


class LocalModel(nn.Module):
    def __init__(self, in_dim, hist_len, node_gru_hidden_dim, city_num, batch_size, device):
        super(LocalModel, self).__init__()

        self.num_nodes = city_num
        self.batch_size = batch_size
        self.device = device
        self.hid_dim = node_gru_hidden_dim
        self.in_dim = in_dim
        self.hist_len = hist_len

        self.gru_cell = GRUCell(self.in_dim, self.hid_dim).to(self.device)
        self.node_mlp = Linear(self.hid_dim, 1).to(self.device)

        self.reset_hn()

    def expected_cn_len(self, transfer_model):
        return transfer_model.hist_len

    def reset_hn(self):
        h0 = torch.zeros(self.batch_size * self.num_nodes, self.hid_dim).to(self.device)
        self.hn = h0

    def forward(self, cn, i, feature):
        node_in = torch.cat([feature[:, self.hist_len + i], cn[:, 0]], dim=2)
        node_in = node_in.contiguous()
        self.hn = self.gru_cell(node_in.to(self.device), self.hn.to(self.device))
        hn_reshaped = self.hn.view(self.batch_size, self.num_nodes, self.hid_dim)
        
        return self.node_mlp(hn_reshaped)