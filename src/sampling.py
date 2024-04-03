import pandas as pd
import numpy as np
import torch

def random_sampling(data_loader, sample_batch_size, device):
    # aggregate bags
    reveal_df = pd.DataFrame(columns=["bag", "index"])
    for X, y, bags, bag_prop in data_loader:
        X, y = X[0].to(device), y[0].to(device)
        idx = np.arange(len(y.cpu().numpy()))
        bags = bags[0].cpu().numpy()
        bag_df = pd.DataFrame({"bag" : bags, "index" : idx})
        reveal_df = pd.concat([reveal_df, bag_df])
        
    return reveal_df.sample(n=sample_batch_size).reset_index(drop=True)

def entropy_sampling(model, data_loader, revealed_labels, sample_batch_size, device):
    # compute entropy
    score_df = pd.DataFrame(columns=["bag", "index", "score"])
    model.eval()
    with torch.no_grad():
        for X, y, bags, bag_prop in data_loader:
            X, y = X[0].to(device), y[0].to(device)
            # working with both dllp + al and only al
            try:
                batch_avg, outputs = model(X)
            except:
                outputs = model(X)
            scores = torch.sum(torch.special.entr(torch.exp(outputs)), axis=1).cpu().numpy()
            idx = np.arange(len(scores))
            bags = bags[0].cpu().numpy()
            bag_df = pd.DataFrame({"bag" : bags, "index" : idx, "score" : scores})
            score_df = pd.concat([score_df, bag_df])

    # get subset not revelead  
    intersec_df = score_df[~(score_df["bag"].isin(revealed_labels["bag"]) & score_df["index"].isin(revealed_labels["index"]))].sort_values(by="score", ascending=False)

    # reveal labels n
    revealed_labels = pd.concat([revealed_labels, intersec_df.head(sample_batch_size)[["bag", "index"]]])
    return revealed_labels.reset_index(drop=True)

def kld_entropy_sampling(model, data_loader, revealed_labels, sample_batch_size, kld_distances, device):
    # compute entropy
    score_df = pd.DataFrame(columns=["bag", "index", "score"])
    model.eval()
    with torch.no_grad():
        for X, y, bags, bag_prop in data_loader:
            X, y = X[0].to(device), y[0].to(device)
            # working with both dllp + al and only al
            try:
                batch_avg, outputs = model(X)
            except:
                outputs = model(X)
            scores = torch.sum(torch.special.entr(torch.exp(outputs)), axis=1).cpu().numpy() * kld_distances[bags[0]] # multiply by kld_distance score
            idx = np.arange(len(scores))
            bags = bags[0].cpu().numpy()
            bag_df = pd.DataFrame({"bag" : bags, "index" : idx, "score" : scores})
            score_df = pd.concat([score_df, bag_df])

    # get subset not revelead  
    intersec_df = score_df[~(score_df["bag"].isin(revealed_labels["bag"]) & score_df["index"].isin(revealed_labels["index"]))].sort_values(by="score", ascending=False)

    # reveal labels n
    revealed_labels = pd.concat([revealed_labels, intersec_df.head(sample_batch_size)[["bag", "index"]]])
    return revealed_labels.reset_index(drop=True)