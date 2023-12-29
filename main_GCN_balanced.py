import numpy as np
import random
import copy

import torch
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
import torch.nn.functional as F

import os
from metric import get_metrics, metrics
from model import GCN, balanced_focal_loss

tot_real_table = np.empty([0], dtype=int)
tot_predict_table = np.empty([0], dtype=int)
tot_predict_score = np.empty([0], dtype=float)

def get_val_loss(model, data, device):
    model.eval()
    out = model(data)
    loss = balanced_focal_loss(out[data.test_mask], data.y[data.test_mask], alpha=1.0, reduction="mean")
    model.train()
    return loss.item()

def train(model, data, device, epochs):
    model.train()
    data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    min_epochs = 50
    min_val_loss = 50000.0
    best_model = None

    for epoch in range(epochs):

        out = model(data).to(device)
        optimizer.zero_grad()

        label = data.y[data.train_mask]

        loss = balanced_focal_loss(out[data.train_mask], data.y[data.train_mask], alpha=1.0, reduction="mean")
        loss.backward()
        optimizer.step()

        val_loss = get_val_loss(model, data, device)
        if val_loss < min_val_loss and epoch + 1 > min_epochs:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        if epoch % 1000 == 0:
            _, pred = out.max(dim=1)
            correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
            Acc = correct / int(data.train_mask.sum())
            print('GCN train_Acc: {:.4f}'.format(Acc))
            print('Epoch {:03d} train_loss {:.4f}'.format(epoch, loss.item()))
            print('Epoch {:03d} test_loss {:.4f}'.format(epoch, val_loss))

    print('Optimization Finished!')
    return best_model


def test(model, data):
    model.eval()
    out = model(data)
    #将结果映射为概率
    out = F.softmax(out, dim=1)
    score, pred = out.max(dim=1)

    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    Acc = correct / int(data.test_mask.sum())
    print('GCN test_Acc: {:.4f}'.format(Acc))

    real_table = data.y[data.test_mask]
    predict_table = pred[data.test_mask]
    ans = out[data.test_mask]
    predict_score = ans[:, 1]

    real_table = real_table.cpu().numpy()
    predict_table = predict_table.cpu().numpy()
    predict_score = predict_score.cpu().detach().numpy()

    global tot_real_table, tot_predict_table, tot_predict_score
    tot_real_table = np.append(tot_real_table, real_table)
    tot_predict_table = np.append(tot_predict_table, predict_table)
    tot_predict_score = np.append(tot_predict_score, predict_score)

    return get_metrics(real_table, predict_table, predict_score)

def cross_validation_experiment(data, k_flod, seed, num_classes, epochs):
    num_nodes = data.x.shape[0]
    num_node_features = data.x.shape[1]
    # 对现有结点进行编号和打乱
    random_index = [i for i in range(num_nodes)]
    for j in range(num_nodes):
        random_index[j] %= k_flod
    random.seed(seed)
    # 随机打乱
    random.shuffle(random_index)

    metric = np.zeros((1, 7))
    for i in range(k_flod):
        print("------this is %dth cross validation------" % (i + 1))
        train_mask = torch.ones(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for j in range(num_nodes):
            if random_index[j] == i:
                test_mask[j] = 1
                train_mask[j] = 0

        data.train_mask = train_mask
        data.test_mask = test_mask

        print("data.train_mask: ", type(data.train_mask), data.train_mask.shape)
        print("data.test_mask: ", type(data.test_mask), data.test_mask.shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(num_node_features, num_classes).to(device)

        pos_number = int(sum(y))
        neg_number = y.shape[0] - int(sum(y))

        data.lossweights = torch.tensor([1.0, neg_number * 1.0 / pos_number], dtype=torch.float32)
        print("lossweights shape: ", type(data.lossweights), data.lossweights.shape, data.lossweights)
        model = train(model, data, device, epochs)

        metric_tmp = test(model, data)
        print("[AUPR, AUC, f1, acc, recall, pre]")
        print("%d-flod metric: " % (i + 1), metric_tmp)
        metric += metric_tmp
    metric = np.array(metric / k_flod)
    return metric


if __name__ == "__main__":

    seed = 99
    processed_dir = '/'
    num_classes = 2
    epochs = 15001
    k_flod = 10

    y = np.loadtxt('Y.csv', delimiter=',')

    x = np.loadtxt('X_PCA512.csv', delimiter=',')
    edge_index_PN = np.loadtxt('NN3_1_1_balance_edge_index.csv', delimiter=',')

    X = torch.tensor(x, dtype=torch.float)
    Y = torch.tensor(y, dtype=torch.long)

    PN_edge_str = [each[0] for each in edge_index_PN]
    PN_edge_end = [each[1] for each in edge_index_PN]

    Edge_index_PN = torch.tensor([PN_edge_str, PN_edge_end], dtype=torch.long)

    data_PN = Data(x=X, edge_index=Edge_index_PN, y=Y)
    # torch.save(data_KNN, os.path.join(processed_dir, f'data_PN.pt'))

    print("data len:", len(data_PN))
    print("data.x shape: ", data_PN.x.shape)

    print("data.y shape: ", data_PN.y.shape)
    print("data.edge_index shape: ", data_PN.edge_index.shape)

    # return [AUPR, AUC, f1, acc, recall, pre]
    metric = cross_validation_experiment(data_PN, k_flod, seed, num_classes, epochs)
    print("average metric: ", metric)

    m = get_metrics(tot_real_table, tot_predict_table, tot_predict_score)
    print("tot metric: ", m)


