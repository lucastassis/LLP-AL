import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mlp(model, optimizer, n_epochs, loss_fc, data_loader, device):  
    model.train()
    for t in range(n_epochs):
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)  
            # compute outputs
            outputs = model(X)    
            # compute loss and backprop
            optimizer.zero_grad()
            loss = loss_fc(outputs, y)
            loss.backward()
            optimizer.step()

def train_active_mlp(model, optimizer, n_epochs, loss_act, data_loader, reveal_labels, device):  
    model.train()
    for i in range(n_epochs):
        for X, y, bags, bag_prop in data_loader:
            # prepare bag data
            X, y, bags, bag_prop = X[0].to(device), y[0].to(device), bags[0].to(device), bag_prop[0].to(device)
            # compute output                
            outputs = model(X) 
            # get index in bags
            idx = reveal_labels[reveal_labels["bag"] == bags[0]]["index"].tolist()
            # compute loss and backprop
            if idx:
                optimizer.zero_grad()
                loss = loss_act(outputs[idx], y[idx])
                loss.backward()
                optimizer.step()

def train_dllp(model, optimizer, n_epochs, loss_fc, data_loader, device):
    model.train()    
    for i in range(n_epochs):
        for X, y, bags, bag_prop in data_loader:
            # prepare bag data
            X, bag_prop = X[0].to(device), bag_prop[0].to(device)
            # compute outputs
            batch_avg, outputs = model(X) 
            # compute loss and backprop
            optimizer.zero_grad()
            loss = loss_fc(batch_avg, bag_prop)
            loss.backward()
            optimizer.step()    

def train_dllp_active(model, optimizer, n_epochs, loss_batch, loss_act, c, data_loader, reveal_labels, device):
    losses = []
    model.train()
    for i in range(n_epochs):
        loss_epoch = 0
        for X, y, bags, bag_prop in data_loader:
            # prepare bag data
            X, y, bags, bag_prop = X[0].to(device), y[0].to(device), bags[0].to(device), bag_prop[0].to(device)
            # compute output                
            batch_avg, outputs = model(X)     
            # get index in bags
            idx = reveal_labels[reveal_labels["bag"] == bags[0]]["index"].tolist()
            # compute loss and backprop
            if idx:
                optimizer.zero_grad()
                loss_b = loss_batch(batch_avg, bag_prop)
                loss_a = loss_act(outputs[idx], y[idx])
                loss = loss_b + c * loss_a
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss_b = loss_batch(batch_avg, bag_prop)
                loss = loss_b
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()
        losses.append(loss_epoch / len(data_loader))
    return losses

    




    


