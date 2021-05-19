import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from graph import Graph
from dataset import HazeData

from model.MLP import MLP
from model.LSTM import LSTM
from model.GRU import GRU
from model.GC_LSTM import GC_LSTM
from model.nodesFC_GRU import nodesFC_GRU
from model.PM25_GNN import PM25_GNN
from model.PM25_GNN_nosub import PM25_GNN_nosub
from model.SplitGNN_3 import SplitGNN_3
from model.SplitGNN_4 import LocalModel, OracleModel, TransferModel
from model.SplitGNN_5 import SplitGNN_5

import arrow
import torch
import torch.autograd.profiler as profiler
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil
import wandb

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device(config['device'][os.uname().nodename]['torch_device'])

graph = Graph()
city_num = graph.node_num

batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
step_lr_step_size = config['train']['step_lr']['step_size']
step_lr_gamma = config['train']['step_lr']['gamma']

results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
node_gru_hidden_dim = config['node_gru_hidden_dim']
edge_gru_hidden_dim = config['edge_gru_hidden_dim']
edge_mlp_hidden_dim = config['edge_mlp_hidden_dim']

exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()

use_pm25 = hist_len > 0

if not use_pm25:
    print('Not using PM25 measurements as inputs')

train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
in_dim = train_data.feature.shape[-1] + (train_data.pm25.shape[-1] * int(use_pm25))
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std


def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi, pod, far


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'Device: %s\n' % os.uname().nodename + \
                'Device config: %s\n' % config['device'][os.uname().nodename] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info


def get_model():
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'SplitGNN_3':
        return SplitGNN_3(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std, node_gru_hidden_dim, edge_gru_hidden_dim, edge_mlp_hidden_dim)
    elif exp_model == 'SplitGNN_4':
        node_module = OracleModel(node_gru_hidden_dim, city_num, batch_size, device)
        return TransferModel(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std, edge_gru_hidden_dim, edge_mlp_hidden_dim, node_module)
    elif exp_model == 'SplitGNN_5':
        return SplitGNN_5(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std, node_gru_hidden_dim, edge_gru_hidden_dim, edge_mlp_hidden_dim)
    else:
        raise Exception('Wrong model name!')


def train(train_loader, model, optimizer, model_input="hist"):
    """Trains the model

    Args:
        train_loader (DataLoader): load training data
        model (Module): model to train
        optimizer: optimizer to use
        model_input (str, optional): "hist" for passing historical pm25 values to the model or
        "label" to pass true labels (for pre-training). Defaults to "hist".

    Returns:
        float: train loss
    """
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)

        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]

        if model_input == "hist":
            out = model(pm25_hist, feature)
        elif model_input == "label":
            out = model(pm25, feature)
        else:
            raise RuntimeError(f"Unknown model input type '{model_input}'")

        if hasattr(model, 'returns_r') and model.returns_r:
            pm25_pred, _ = out
        else:
            pm25_pred = out

        loss = criterion(pm25_pred, pm25_label)
        wandb.log({'train_loss': loss})
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx + 1
    return train_loss


def val(val_loader, model, model_input="hist"):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(tqdm(val_loader)):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]

        if model_input == "hist":
            out = model(pm25_hist, feature)
        elif model_input == "label":
            out = model(pm25, feature)
        else:
            raise RuntimeError(f"Unknown model input type '{model_input}'")

        if hasattr(model, 'returns_r') and model.returns_r:
            pm25_pred, _ = out
        else:
            pm25_pred = out

        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item()

    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, model, model_input="hist"):
    model.eval()
    predict_list = []
    R_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]

        if model_input == "hist":
            out = model(pm25_hist, feature)
        elif model_input == "label":
            out = model(pm25, feature)
        else:
            raise RuntimeError(f"Unknown model input type '{model_input}'")

        if hasattr(model, 'returns_r') and model.returns_r:
            pm25_pred, R = out
        else:
            pm25_pred, R = out, torch.empty(0)

        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())
        R_list.append(R.cpu().detach().numpy())

    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0
    R_epoch = np.concatenate(R_list, axis=0)

    return test_loss, predict_epoch, label_epoch, time_epoch, R_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def pre_train(model, exp_model_id, exp_model_group_id, train_loader, val_loader, test_loader):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    if step_lr_step_size is not None and step_lr_gamma is not None:
        scheduler = StepLR(optimizer, step_size=step_lr_step_size, gamma=step_lr_gamma)
    else:
        scheduler = None

    run = wandb.init(reinit=True, name=exp_model_id + "_pre-train", entity="split-sources", project="split-pollution-sources", config=config)
    wandb.config.exp_group = exp_model_group_id
    wandb.config.model = type(model.node_module).__name__
    # wandb.watch(model)

    val_loss_min = 100000
    best_epoch = 0

    train_loss_, val_loss_ = 0, 0

    for epoch in range(epochs):
        print('\nTrain epoch %s:' % (epoch))

        train_loss = train(train_loader, model, optimizer, model_input="label")
        val_loss = val(val_loader, model, model_input="label")

        if scheduler is not None:
            scheduler.step()

        print('train_loss: %.4f' % train_loss)
        print('val_loss: %.4f' % val_loss)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})

        if epoch - best_epoch > early_stop:
            break

        if val_loss < val_loss_min:
            val_loss_min = val_loss
            best_epoch = epoch
            print('Minimum val loss!!!')

            test_loss, predict_epoch, label_epoch, time_epoch, R_epoch = test(test_loader, model, model_input="label")
            train_loss_, val_loss_ = train_loss, val_loss
            rmse, mae, csi, pod, far = get_metric(predict_epoch, label_epoch)
            print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))
            wandb.log({'epoch': epoch, 'test_loss': test_loss, 'rmse': rmse, 'mae': mae, 'csi': csi, 'pod': pod, 'far': far})

    run.finish()

    print("Done pre-training")

    node_module = LocalModel(model.in_dim, model.hist_len, model.node_module.hid_dim, model.node_module.num_nodes, model.node_module.batch_size, model.node_module.device)
    model.node_module = node_module

    return model


def main():
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYY-MM-DD_HH-mm-ss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        print(str(model))
        
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable_params}")

        exp_model_group_id = os.path.join('%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time))
        exp_model_id = os.path.join(exp_model_group_id, '%02d' % exp_idx)

        if isinstance(model, TransferModel):
            model = pre_train(model, exp_model_id, exp_model_group_id, train_loader, val_loader, test_loader)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        if step_lr_step_size is not None and step_lr_gamma is not None:
            scheduler = StepLR(optimizer, step_size=step_lr_step_size, gamma=step_lr_gamma)
        else:
            scheduler = None

        run = wandb.init(reinit=True, name=exp_model_id, entity="split-sources", project="split-pollution-sources", config=config)
        wandb.config.exp_group = exp_model_group_id
        wandb.config.model = model_name
        wandb.watch(model)

        exp_model_dir = os.path.join(results_dir, exp_model_id)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, model, optimizer)
            val_loss = val(val_loader, model)

            if scheduler is not None:
                scheduler.step()

            if hasattr(model, 'alpha'):
                wandb.log({'alpha': model.alpha.cpu()})

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})

            if epoch - best_epoch > early_stop:
                break

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                test_loss, predict_epoch, label_epoch, time_epoch, R_epoch = test(test_loader, model)
                train_loss_, val_loss_ = train_loss, val_loss
                rmse, mae, csi, pod, far = get_metric(predict_epoch, label_epoch)
                print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))
                wandb.log({'epoch': epoch, 'test_loss': test_loss, 'rmse': rmse, 'mae': mae, 'csi': csi, 'pod': pod, 'far': far})

                if save_npy:
                    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
                    np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)
                    np.save(os.path.join(exp_model_dir, 'R.npy'), R_epoch)

        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))

        run.finish()

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list))

    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)


if __name__ == '__main__':
    main()
