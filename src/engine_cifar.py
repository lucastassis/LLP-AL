import torch
import pandas as pd
import numpy as np
from net import CNNBatchAvg, CNN
from sampling import entropy_sampling, kld_entropy_sampling, random_sampling 
from train import train_mlp, train_active_mlp, train_dllp, train_dllp_active
from image_loader import ImageLLPDataset, ImageDataset
from eval import eval_dllp, eval_mlp
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_cnn_mlp(X_train, X_test, y_train, y_test):
    # load dataset and create dataloader
    train_dataset = ImageDataset(data=X_train, targets=y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512)
    test_dataset = ImageDataset(data=X_test, targets=y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # define model
    model = CNN(n_classes=len(np.unique(y_test)))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fc = torch.nn.NLLLoss()
    
    # train and eval model
    train_mlp(model=model, optimizer=optimizer, n_epochs=100, loss_fc=loss_fc, data_loader=train_dataloader, device=device)
    metrics = eval_mlp(model=model, data_loader=test_dataloader, device=device)
    print(metrics)
    return metrics

def run_cnn_active_mlp(X_train, X_test, y_train, y_test, bags_train, bags_test, n_reveal, sample_batch_size):
    # load dataset and create dataloader
    train_dataset = ImageLLPDataset(data=X_train, targets=y_train, bags=bags_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    test_dataset = ImageDataset(data=X_test, targets=y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    metrics = []
    # reveal first label at random
    revealed_labels = random_sampling(data_loader=train_dataloader, sample_batch_size=sample_batch_size, device=device)

    for i in range(sample_batch_size, n_reveal[-1], sample_batch_size):
        # define model
        model = CNN(n_classes=len(np.unique(y_test)))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_act = torch.nn.NLLLoss()

        # train model
        train_active_mlp(model=model, optimizer=optimizer, n_epochs=100, loss_act=loss_act, data_loader=train_dataloader, reveal_labels=revealed_labels, device=device)
        
        if i in n_reveal:
            print(len(revealed_labels))
            metrics.append(eval_mlp(model=model, data_loader=test_dataloader, device=device))

        # reveal next label
        revealed_labels = entropy_sampling(model=model, revealed_labels=revealed_labels, data_loader=train_dataloader, sample_batch_size=sample_batch_size, device=device)
    
    # train and eval last model
    # define model
    model = CNN(n_classes=len(np.unique(y_test)))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_act = torch.nn.NLLLoss()

    # train model
    train_active_mlp(model=model, optimizer=optimizer, n_epochs=100, loss_act=loss_act, data_loader=train_dataloader, reveal_labels=revealed_labels, device=device)
    print(len(revealed_labels))
    metrics.append(eval_mlp(model=model, data_loader=test_dataloader, device=device))
    print(metrics)
    return metrics

def run_cnn_dllp(X_train, X_test, y_train, y_test, bags_train, bags_test):
    # load dataset and create dataloader
    train_dataset = ImageLLPDataset(data=X_train, targets=y_train, bags=bags_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    test_dataset = ImageLLPDataset(data=X_test, targets=y_test, bags=bags_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # define model
    model = CNNBatchAvg(n_classes=len(np.unique(y_test)))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fc = torch.nn.KLDivLoss(reduction="batchmean")
    
    # train and eval model
    train_dllp(model=model, optimizer=optimizer, n_epochs=100, loss_fc=loss_fc, data_loader=train_dataloader, device=device)
    metrics = eval_dllp(model=model, data_loader=test_dataloader, device=device)
    print(metrics)
    return metrics

def run_cnn_rand_sampling(X_train, X_test, y_train, y_test, bags_train, bags_test, n_reveal, c):
    # load dataset and create dataloader
    train_dataset = ImageLLPDataset(data=X_train, targets=y_train, bags=bags_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    test_dataset = ImageLLPDataset(data=X_test, targets=y_test, bags=bags_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # define model
    model = CNNBatchAvg(n_classes=len(np.unique(y_test)))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_batch = torch.nn.KLDivLoss(reduction="batchmean")
    loss_act = torch.nn.NLLLoss()
    
    # reveal labels
    revealed_labels = random_sampling(data_loader=train_dataloader, sample_batch_size=n_reveal, device=device)
    
    # train and eval model
    train_dllp_active(model=model, 
                      optimizer=optimizer, n_epochs=100, loss_batch=loss_batch, loss_act=loss_act, c=c, data_loader=train_dataloader, reveal_labels=revealed_labels, device=device)
    metrics = eval_dllp(model=model, data_loader=test_dataloader, device=device)

    return metrics

def run_cnn_entr_sampling(X_train, X_test, y_train, y_test, bags_train, bags_test, n_reveal, c, sample_batch_size):
    # load dataset and create dataloader
    train_dataset = ImageLLPDataset(data=X_train, targets=y_train, bags=bags_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    test_dataset = ImageLLPDataset(data=X_test, targets=y_test, bags=bags_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    metrics = []
    # reveal first label at random
    revealed_labels = random_sampling(data_loader=train_dataloader, sample_batch_size=sample_batch_size, device=device)

    for i in range(sample_batch_size, n_reveal[-1], sample_batch_size):
        # define model
        model = CNNBatchAvg(n_classes=len(np.unique(y_test)))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_batch = torch.nn.KLDivLoss(reduction="batchmean")
        loss_act = torch.nn.NLLLoss()

        # train model
        train_dllp_active(model=model, optimizer=optimizer, n_epochs=100, loss_batch=loss_batch, loss_act=loss_act, c=c, data_loader=train_dataloader, reveal_labels=revealed_labels, device=device)
        
        if i in n_reveal:
            print(len(revealed_labels))
            metrics.append(eval_dllp(model=model, data_loader=test_dataloader, device=device))

        # reveal next label
        revealed_labels = entropy_sampling(model=model, revealed_labels=revealed_labels, data_loader=train_dataloader, sample_batch_size=sample_batch_size, device=device)
    
    # train and eval last model
    model = CNNBatchAvg(n_classes=len(np.unique(y_test)))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_batch = torch.nn.KLDivLoss(reduction="batchmean")
    loss_act = torch.nn.NLLLoss()
    train_dllp_active(model=model, optimizer=optimizer, n_epochs=100, loss_batch=loss_batch, loss_act=loss_act, c=c, data_loader=train_dataloader, reveal_labels=revealed_labels, device=device)
    metrics.append(eval_dllp(model=model, data_loader=test_dataloader, device=device))

    return metrics

def run_cnn_kld_entr_sampling(X_train, X_test, y_train, y_test, bags_train, bags_test, n_reveal, c, sample_batch_size):
    # load dataset and create dataloader
    train_dataset = ImageLLPDataset(data=X_train, targets=y_train, bags=bags_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    test_dataset = ImageLLPDataset(data=X_test, targets=y_test, bags=bags_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # compute kld_score
    dataset_prop = torch.tensor(pd.Series(y_train).value_counts().sort_index().to_numpy() / len(y_train))

    # compute kld_score for bag
    bag_target_df = pd.DataFrame({"bag" : bags_train, "target" : y_train}, dtype="category")
    props_list = [bag_target_df[bag_target_df["bag"] == bag]["target"].value_counts().sort_index().to_numpy() / len(bag_target_df[bag_target_df["bag"] == bag]["target"]) for bag in np.sort(bag_target_df["bag"].unique())]
    bags_prop = torch.tensor(np.vstack(props_list))
    kld_distances = np.array([torch.exp(-F.kl_div(bags_prop[i].log(), dataset_prop, reduction="batchmean")).item() for i in range(0, len(bag_target_df["bag"].unique()))])
    norm_kld_distances = kld_distances / np.linalg.norm(kld_distances, ord=1)
    
    metrics = []

    revealed_labels = pd.DataFrame(columns=["bag", "index"])

    for i in range(0, n_reveal[-1], sample_batch_size):
        # define model
        model = CNNBatchAvg(n_classes=len(np.unique(y_test)))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_batch = torch.nn.KLDivLoss(reduction="batchmean")
        loss_act = torch.nn.NLLLoss()

        # train model
        train_dllp_active(model=model, optimizer=optimizer, n_epochs=100, loss_batch=loss_batch, loss_act=loss_act, c=c, data_loader=train_dataloader, reveal_labels=revealed_labels, device=device)
        
        if i in n_reveal:
            print(len(revealed_labels))
            metrics.append(eval_dllp(model=model, data_loader=test_dataloader, device=device))

        # reveal next label
        revealed_labels = kld_entropy_sampling(model=model, revealed_labels=revealed_labels, data_loader=train_dataloader, 
                                               sample_batch_size=sample_batch_size, kld_distances=norm_kld_distances, device=device)
    
    # train and eval last model
    model = CNNBatchAvg(n_classes=len(np.unique(y_test)))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_batch = torch.nn.KLDivLoss(reduction="batchmean")
    loss_act = torch.nn.NLLLoss()
    train_dllp_active(model=model, optimizer=optimizer, n_epochs=100, loss_batch=loss_batch, loss_act=loss_act, c=c, data_loader=train_dataloader, reveal_labels=revealed_labels, device=device)
    metrics.append(eval_dllp(model=model, data_loader=test_dataloader, device=device))
   
    return metrics