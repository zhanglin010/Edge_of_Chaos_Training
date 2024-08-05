import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils
from loggingreporter import LoggingReporter
import plot_figure2
# import plot_figure4_5
# import plot_figure6


class SimpleNN(nn.Module):
    def __init__(self, args):
        super(SimpleNN, self).__init__()

        if args.activation_func == 'relu':
            self.activation = nn.ReLU()
        elif args.activation_func == 'tanh':
            self.activation = nn.Tanh()

        self.fc1 = nn.Linear(args.input_shape, args.input_shape, bias=False)
        # self.fc2 = nn.Linear(args.input_shape, args.input_shape, bias=False)
        self.fc3 = nn.Linear(args.input_shape, 10)

        nn.init.normal_(self.fc1.weight, mean=(0)/(args.input_shape), 
                        std=1/(2*np.sqrt(args.input_shape)))

        self.regularizer = nn.Parameter(torch.tensor(args.weight_decay))

    def forward(self, x):
        x = self.activation(self.fc1(x))
        # x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


def main(args):
    start_time = time.time()

    if not args.train:
        print('-------------------------')
        print('| Only plot the figures |')
        print('-------------------------')

    args.log_epochs = np.arange(args.epochs) # Log weights at these epochs
    args.arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(
        args.dataset, 
        args.optimizer, 
        args.activation_func, 
        args.epochs, 
        int(args.lr*10000), 
        int(args.batch_size), 
        int(args.momentum*1000), 
        int(args.beta_1*1000), 
        int(args.beta_2*10000000), 
        int(args.weight_decay*1000000)
        )
    print(f'arguments={args.arguments}')

    args.save_weights_dir = f'./rawdata/weights/{args.arguments}'
    args.save_losses_dir = f'./rawdata/losses/{args.arguments}'
    args.save_scores_dir = f'./rawdata/scores/{args.arguments}'

    (x_train, y_train), (x_test, y_test) = utils.load_data(args.dataset)
    args.input_shape = x_train.shape[1]
    # print(f'input_shape={args.input_shape}') # 784
    
    for num_repeat in range(args.num_repeats):
        if not args.train:
            break # Only plot the figures

        args.save_weights_dir_repeat = args.save_weights_dir + f'/{num_repeat}'
        args.save_losses_dir_repeat = args.save_losses_dir + f'/{num_repeat}'
        args.save_scores_dir_repeat = args.save_scores_dir + f'/{num_repeat}'

        model = SimpleNN(args)
        model = model.to(device)
        
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                                  momentum=args.momentum, 
                                  weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                                   betas=(args.beta_1, args.beta_2), 
                                   weight_decay=args.weight_decay)

        metric_loss = nn.CrossEntropyLoss()

        # Prepare data loaders
        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), 
                                      torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), 
                                     torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        reporter = LoggingReporter(args, model, x_train, y_train, x_test, y_test)

        # Training loop
        for epoch in range(args.epochs):
            # print('--------------------------------------------------')
            print(f'num_repeat={num_repeat}, epoch={epoch}')
            reporter.on_epoch_begin(epoch)
            model.train()

            train_loss = 0
            train_correct = 0
            total_train = 0

            # Train the model
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = metric_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_y).sum().item()
                total_train += batch_y.size(0)

            train_loss /= len(train_loader)
            train_acc = train_correct / total_train

            model.eval()
            val_loss = 0
            val_correct = 0
            total_val = 0

            # Validate the model
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_x)
                    loss = metric_loss(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_y).sum().item()
                    total_val += batch_y.size(0)

            val_loss /= len(test_loader)
            val_acc = val_correct / total_val

            # Log the results
            reporter.on_epoch_end(epoch, train_loss, val_loss, train_acc, val_acc)
        
        reporter.on_train_end()

        # Clear session
        torch.cuda.empty_cache()

    # Plotting figures
    plot_figure2.plot_weights_path(args)
    plot_figure2.plot_loss_acc(args)
    plot_figure2.plot_weights_var_mean(args)

    end_time = time.time()
    delta_t = end_time - start_time
    print(f'Running time: {int(delta_t//60)} min {round((10*delta_t)%600)/10} s')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser(description='Asymptotic stability study of SGD')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        help="dataset: {'mnist', 'kmnist', 'emnist/mnist', 'fashion_mnist'}")
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer used for training')
    parser.add_argument('--activation-func', type=str, default='tanh',
                        help='activation function for hidden layers')
    parser.add_argument('--epochs', default=11, type=int, metavar='N',
                        help='number of total epochs to run, should > 3')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='batch size for training')
    parser.add_argument('--lr', '--learning-rate', default=0.02, type=float, metavar='LR', 
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                        help='momentum for SGD')
    parser.add_argument('--beta-1', default=0.9, type=float, metavar='M',
                        help='beta_1 in Adam')
    parser.add_argument('--beta-2', default=0.999, type=float, metavar='M',
                        help='beta_2 in Adam')
    parser.add_argument('--weight-decay', default=0., type=float, metavar='W', 
                        help='weight decay (default: 0)', dest='weight_decay')
    parser.add_argument('--num-repeats', type=int, default=1,
                        help='number of simulation repeats')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model or not')
    args = parser.parse_args()
    
    main(args)
