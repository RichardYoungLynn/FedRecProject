import torch
from torch import nn


class LocalFCFModel(nn.Module):
    def __init__(self, args, features, user_rating, Lambda):
        super(LocalFCFModel, self).__init__()
        self.args=args
        self.Lambda = Lambda
        self.item_factor = None
        self.user_rating = user_rating
        self.mask = self.rating > 0
        self.user_factor = nn.Parameter(nn.init.normal_(torch.empty(1, features), std=0.35), requires_grad=True)
        self.history_gradients = []

    def forward(self, item_factor):
        return self.user_factor.mm(item_factor)

    def calculate_loss(self):
        rmse = self.calculate_rmse()
        regularization = self.Lambda * (torch.norm(self.user_factor)**2 +
                                        (torch.norm(self.item_factor, dim=0)*self.mask)**2)
        return rmse + regularization.sum()

    def set_item_factor(self, item_factor):
        self.item_factor = item_factor.clone().detach().to(self.args.device)
        self.item_factor.requires_grad_(True)

    def calculate_rmse(self):
        return (((self.forward(self.item_factor) - self.user_rating)*self.mask)**2).sum()

    def get_user_factor(self):
        return self.user_factor.detach().cpu()

    def add_his_grad(self, gradient):
        self.history_gradients.append(gradient.detach().cpu().numpy())


class ServerFCFModel(nn.Module):
    def __init__(self, args, item_num, features, device=None):
        super(ServerFCFModel, self).__init__()
        self.args = args
        self.device = device
        self.features = features
        self.item_factor = nn.init.normal_(torch.empty(self.features, item_num), std=0.35).to(self.device)

    def update(self, gradient):
        gradient = gradient.to(self.device)
        self.item_factor = self.item_factor - self.args.lr * gradient

    def get_item_factor(self):
        return self.item_factor.detach()

    def forward(self):
        pass