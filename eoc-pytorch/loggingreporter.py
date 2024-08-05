import os
import pickle
import numpy as np
import torch
import torch.nn as nn

class LoggingReporter:
    """Save the activations to files after some epochs.

    Args:
        args: configuration options dictionary
        x_train: training data
        y_train: training labels
        x_test: test data
        y_test: test labels
    """

    def __init__(self, args, model, x_train, y_train, x_test, y_test, *kargs, **kwargs):
        # super(LoggingReporter, self).__init__(*kargs, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.model = model.to(self.device)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.losses = {'train': [], 'val': []}
        self.W0_init = self.get_weights()

        # Ensure directories exist
        os.makedirs(self.args.save_weights_dir_repeat, exist_ok=True)
        os.makedirs(self.args.save_losses_dir_repeat, exist_ok=True)
        os.makedirs(self.args.save_scores_dir_repeat, exist_ok=True)
    
    def get_weights(self):
        return [param.data.cpu().numpy() for param in self.model.parameters()]

    def save_weights(self, epoch):
        weights = self.get_weights()
        W_diff = np.linalg.norm(weights[1] - self.W0_init[1])
        results = {'W0s': weights[1], 'W_diffs': W_diff}
        fname = '{}/epoch{:05d}'.format(self.args.save_weights_dir_repeat, epoch)
        with open(fname, 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    def on_epoch_begin(self, epoch):
        if epoch in self.args.log_epochs:
            self.save_weights(epoch)
        if epoch == 0: # Save initial losses
            self.losses['train'].append(self.evaluate(self.x_train, self.y_train))
            self.losses['val'].append(self.evaluate(self.x_test, self.y_test))

    def on_epoch_end(self, epoch, train_loss, val_loss, train_acc, val_acc):
        del epoch
        self.losses['train'].append([train_loss, train_acc])
        self.losses['val'].append([val_loss, val_acc])

    def on_train_end(self):
        """Save training losses and scores to file"""
        fname = '{}/losses'.format(self.args.save_losses_dir_repeat)
        with open(fname, 'wb') as f:
            pickle.dump(self.losses, f, pickle.HIGHEST_PROTOCOL)

        scores = self.evaluate(self.x_test, self.y_test)
        fname = '{}/scores'.format(self.args.save_scores_dir_repeat)
        with open(fname, 'wb') as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)

    def evaluate(self, x, y):
        self.model.eval()

        # Ensure inputs are PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)
        return loss, accuracy
