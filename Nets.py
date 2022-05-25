import torch
from torch import nn


class NCF(nn.Module):
    def __init__(self, args):
        super(NCF, self).__init__()

        self.MF_Embedding_User = nn.Embedding(num_embeddings=args.candidate_num, embedding_dim=args.MF_latent_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=args.num_items, embedding_dim=args.MF_latent_dim)

        self.MLP_Embedding_User = nn.Embedding(num_embeddings=args.candidate_num, embedding_dim=args.MLP_latent_dim)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=args.num_items, embedding_dim=args.MLP_latent_dim)
        self.fc1 = nn.Linear(int(args.MLP_latent_dim*2), args.MLP_latent_dim)
        self.fc2 = nn.Linear(args.MLP_latent_dim, int(args.MLP_latent_dim/2))
        self.fc3 = nn.Linear(int(args.MLP_latent_dim/2), int(args.MLP_latent_dim/4))

        self.output = nn.Linear(args.MF_latent_dim+int(args.MLP_latent_dim/4), 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, user_input, item_input):
        MF_User_Vector = self.MF_Embedding_User(user_input)
        # print(user_input.shape)
        # print(user_input)
        MF_Item_Vector = self.MF_Embedding_Item(item_input)
        # MF_User_Vector = torch.flatten(MF_User_Vector)
        # MF_Item_Vector = torch.flatten(MF_Item_Vector)
        mf_vector = torch.mul(MF_User_Vector, MF_Item_Vector)

        MLP_User_Vector = self.MLP_Embedding_User(user_input)
        MLP_Item_Vector = self.MLP_Embedding_Item(item_input)
        # MLP_User_Vector = torch.flatten(MLP_User_Vector)
        # MLP_Item_Vector = torch.flatten(MLP_Item_Vector)
        mlp_vector = torch.cat((MLP_User_Vector, MLP_Item_Vector), 1)
        mlp_vector = self.fc1(mlp_vector)
        mlp_vector = self.relu(mlp_vector)
        mlp_vector = self.fc2(mlp_vector)
        mlp_vector = self.relu(mlp_vector)
        mlp_vector = self.fc3(mlp_vector)
        mlp_vector = self.relu(mlp_vector)

        predict_vector = torch.cat((mf_vector, mlp_vector), 1)
        prediction = self.output(predict_vector)
        prediction = self.sigmoid(prediction)
        prediction = torch.flatten(prediction)
        return prediction


# class GMF(nn.Module):
#     def __init__(self, args):
#         super(GMF, self).__init__()
#
#         self.MF_Embedding_User = nn.Embedding(num_embeddings=args.participant_num, embedding_dim=args.MF_latent_dim)
#         self.MF_Embedding_Item = nn.Embedding(num_embeddings=args.num_items, embedding_dim=args.MF_latent_dim)
#         self.fc = nn.Linear(dim_in, dim_hidden)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, user_input, item_input):
#         MF_User_Vector = self.MF_Embedding_User(user_input)
#         MF_Item_Vector = self.MF_Embedding_Item(item_input)
#
#         MF_User_Vector = torch.flatten(MF_User_Vector)
#         MF_Item_Vector = torch.flatten(MF_Item_Vector)
#
#         predict_vector = torch.mul(MF_User_Vector, MF_Item_Vector)
#         prediction = self.fc(predict_vector)
#         prediction = self.sigmoid(prediction)
#         return prediction
#
#
# class MLP(nn.Module):
#     def __init__(self, args):
#         super(MLP, self).__init__()
#
#         self.MLP_Embedding_User = nn.Embedding(num_embeddings=args.participant_num, embedding_dim=args.MLP_latent_dim)
#         self.MLP_Embedding_Item = nn.Embedding(num_embeddings=args.num_items, embedding_dim=args.MLP_latent_dim)
#         self.fc1 = nn.Linear(dim_in, dim_hidden)
#         self.fc2 = nn.Linear(dim_hidden, dim_hidden)
#         self.fc3 = nn.Linear(dim_hidden, dim_out)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, user_input, item_input):
#         MLP_User_Vector = self.MLP_Embedding_User(user_input)
#         MLP_Item_Vector = self.MLP_Embedding_Item(item_input)
#
#         MLP_User_Vector = torch.flatten(MLP_User_Vector)
#         MLP_Item_Vector = torch.flatten(MLP_Item_Vector)
#
#         predict_vector = torch.cat((MLP_User_Vector, MLP_Item_Vector), 0)
#         prediction = self.fc1(predict_vector)
#         prediction = self.relu(prediction)
#         prediction = self.fc2(predict_vector)
#         prediction = self.relu(prediction)
#         prediction = self.fc3(predict_vector)
#         prediction = self.sigmoid(prediction)
#         return prediction


class LocalFCFModel(nn.Module):
    def __init__(self, args, user_rating):
        super(LocalFCFModel, self).__init__()
        self.args = args
        self.item_factor = None
        self.user_rating = user_rating
        self.mask = self.user_rating > 0
        self.user_factor = nn.Parameter(nn.init.normal_(torch.empty(1, args.feature_num), std=0.35), requires_grad=True)

    def forward(self, item_factor):
        return self.user_factor.mm(item_factor)

    def calculate_loss(self):
        rmse = self.calculate_rmse()
        regularization = self.args.Lambda * (torch.norm(self.user_factor) ** 2 +
                                             (torch.norm(self.item_factor, dim=0) * self.mask) ** 2)
        return rmse + regularization.sum()

    def set_item_factor(self, item_factor):
        self.item_factor = item_factor.clone().detach().to(self.args.device)
        self.item_factor.requires_grad_(True)

    def calculate_rmse(self):
        return (((self.forward(self.item_factor) - self.user_rating) * self.mask) ** 2).sum()

    def get_item_factor(self):
        return self.item_factor


class ServerFCFModel(nn.Module):
    def __init__(self, args, item_num=1682):
        super(ServerFCFModel, self).__init__()
        self.args = args
        self.item_factor = nn.init.normal_(torch.empty(args.feature_num, item_num), std=0.35).to(args.device)

    def update(self, gradient):
        gradient = gradient.to(self.args.device)
        self.item_factor = self.item_factor - self.args.server_lr * gradient

    def get_item_factor(self):
        return self.item_factor.detach()

    def forward(self):
        pass
