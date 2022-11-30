import torch
from torch import nn, optim, no_grad
import numpy as np
import torchvision
import torchvision.transforms as tt
from tqdm import tqdm

import os
import matplotlib.pyplot as plt

import torch.utils.model_zoo as model_zoo
from IPython import embed
from collections import OrderedDict


######

def main_exp(epochs):

	os.getcwd()
	data_path = f"{os.getcwd()}/data/"

	train_dataset = torchvision.datasets.CIFAR10(
	    root=data_path, train=True, download=True,
	    transform=tt.Compose([tt.ToTensor(),
	        #tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	    ]))
	test_and_val_dataset = torchvision.datasets.CIFAR10(
	    root=data_path, train=False, download=True,
	    transform=tt.Compose([tt.ToTensor(),
	        #tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	    ]))
	print(train_dataset[0])

	classes = [0,1]
	remap = {x:i for i, x in enumerate(classes)}
	idx_train = [i for i, label in enumerate(train_dataset.targets) if label in classes]

	targets_train = [train_dataset.targets[idx_train_] for idx_train_ in idx_train]
	targets_train = torch.tensor([remap[t_] for t_ in targets_train])
	train_dataset.targets = targets_train
	train_dataset.data = train_dataset.data[idx_train]

	idx_test_and_val = [i for i, label in enumerate(test_and_val_dataset.targets) if label in classes]
	targets_test_and_val = [test_and_val_dataset.targets[idx_] for idx_ in idx_test_and_val]
	targets_test_and_val = torch.tensor([remap[t_] for t_ in targets_test_and_val])
	test_and_val_dataset.targets = targets_test_and_val
	test_and_val_dataset.data = test_and_val_dataset.data[idx_test_and_val]




	class Model(object):
	    """
	    Target class is C := {x : f(x) <= 0}. The predict method implements f.
	    """
	    def __init__(self, d=784):
	        self.d = d
	        self.R = None
	        self.L = None

	    def predict(self, X):
	        raise NotImplementedError

	    def grad(self, X):
	        raise NotImplementedError

	    def __repr__(self):
	        return "%s(d=%d)" % (self.__class__.__name__, self.d)



	class CIFAR_(nn.Module):
	    def __init__(self, n_channel, num_classes=10):
	        super(CIFAR_, self).__init__()
	        
	        cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
	        #features = make_layers(cfg, batch_norm=True)
	        layers = []
	        in_channels = 3
	        for i, v in enumerate(cfg):
	            if v == 'M':
	                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
	            else:
	                padding = v[1] if isinstance(v, tuple) else 1
	                out_channels = v[0] if isinstance(v, tuple) else v
	                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
	                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
	                in_channels = out_channels
	        features = nn.Sequential(*layers)
	          
	        assert isinstance(features, nn.Sequential), type(features)
	        self.features = features
	        self.fc1 = nn.Linear(10, 1)
	        self.sigmoid = nn.Sigmoid()
	        self.classifier = nn.Sequential(
	            nn.Linear(8*n_channel, num_classes)
	        )    
	        print(self.features)
	        print(self.classifier)

	    def forward(self, x):
	        x = self.features(x)
	        x = x.view(x.size(0), -1)
	        x = self.classifier(x)
	        x = self.fc1(x)
	        
	        return x


	class CIFAR_ResNet(nn.Module, Model):
	    def __init__(self, d=3*32*32, L=None, R=np.inf,
	                 model_path='http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth'):
	        super(CIFAR_ResNet, self).__init__()
	        self.nb_classes = 10
	        self.model_path = model_path
	        self.model_name = "cifar_resnet"
	        self.network = CIFAR_(128, self.nb_classes)
	        m = model_zoo.load_url(model_path, map_location=torch.device('cpu'))
	        m['fc1.weight'] = torch.rand(1,10)
	        m['fc1.bias'] = torch.rand(1)
	        self.network.load_state_dict(m)
	        self.network.eval()
	        self.d = d
	        self.R = R
	        self.L = L
	        self.sigmoid = nn.Sigmoid()

	    def forward(self, x):
	        x = torch.tensor(x).float()
	        x = x.view(-1, 3, 32, 32)
	        return self.sigmoid(self.network(x))

	    def __repr__(self):
	        return "CIFAR_ResNet(d=%d, regime='fully trained')" % (self.d)

	    def predict(self, X, res_type="prediction"):
	        output = self.forward(X)
	        if res_type == "prediction":
	            real_pred = output.argmax(dim=1, keepdim=True)
	            if real_pred == 0:
	                res = 1
	            else:
	                res = -1
	            return res
	        else:
	            return output

	    def grad(self, X):
	        grad_list = list()
	        for x_ in X:
	            x_ = x_.float()
	            x_.requires_grad = True
	            self(x_).backward()
	            grad_list.append(x_.grad.unsqueeze(0))
	        grads = torch.cat(grad_list)
	        return grads


	model = CIFAR_ResNet()

	def compute_acc(model, dataset):
	    correct = 0
	    with no_grad():
	        for i, (data, target) in enumerate(tqdm(dataset)):
	            data = data.unsqueeze(0).float()
	            target = torch.tensor(target).unsqueeze(0)
	            pred = model(data).argmax(dim=1, keepdim=True)
	            correct += pred.eq(target.view_as(pred)).sum().item()
	    acc = correct / len(dataset)
	    return acc

	def fit(model, epochs, dataset, dataset_val, model_path=f"{os.getcwd()}/trained_models/"):
	    if not os.path.exists(model_path):
	        os.makedirs(model_path)
	    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.01)
	    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
	        optimizer, mode="min", patience=10, verbose=True, factor=0.5)
	    loss_func = nn.BCELoss() #nn.CrossEntropyLoss()
	    
	    loss_history = list()
	    for epoch in range(epochs):
	        tot_loss = list()
	        for i_batch, (x_batch, y_batch) in enumerate(tqdm(dataset)):
	            x_batch = x_batch.unsqueeze(0).float()
	            y_batch = torch.tensor(y_batch).unsqueeze(0).float()
	            optimizer.zero_grad()
	            y_pred = model(x_batch).squeeze(0)
	            loss = loss_func(y_pred, y_batch)
	            loss.backward()
	            optimizer.step()
	            tot_loss.append(loss.item())
	        loss_history.append(np.round(np.mean(tot_loss), 5))
	        print(f"Epoch {epoch}: training loss = {np.round(np.mean(tot_loss), 5)}, train. accuracy = {compute_acc(model, dataset_val)}")

	    torch.save(model.state_dict(), model_path+f"CIFAR10_ResNet_epochs_{epochs}.pt")
	    return model, loss_history



	model_trained, loss_history = fit(model, 3, train_dataset, train_dataset)


if __name__ == "__main__":
	epochs = 50
	main_exp(epochs)

