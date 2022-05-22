import argparse


def args_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--local_epochs', type=int, default=10, help="the number of local epochs")
    parser.add_argument('--client_selection', type=str, default='random', help="the way of selecting participants from candidates")
    parser.add_argument('--model', type=str, default='fcf', help="name of model")
    parser.add_argument('--dataset', type=str, default='movielens', help="name of dataset")
    parser.add_argument('--candidate_num', type=int, default=943, help="number of candidate")
    parser.add_argument('--participant_num', type=int, default=600, help="number of participant in pre-experiment")
    parser.add_argument('--local_batch_size', type=int, default=32, help="local batch size")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--test_batch_size', type=int, default=128, help="test batch size")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    # FCF
    parser.add_argument('--server_lr', type=float, default=0.1, help="server's learning rate")
    parser.add_argument('--client_lr', type=float, default=1e-4, help="client's learning rate")
    parser.add_argument('--Lambda', type=float, default=0.02, help="penalty factor")
    parser.add_argument('--feature_num', type=int, default=5, help="feature in latent factor")

    # NCF
    parser.add_argument('--MF_latent_dim', type=int, default=8, help="feature number in MF latent factor")
    parser.add_argument('--MLP_latent_dim', type=int, default=32, help="feature number in MLP latent factor")
    parser.add_argument('--items_num', type=int, default=5, help="items number")
    parser.add_argument('--topK', type=int, default=10, help="topK")



    args = parser.parse_args()
    return args
