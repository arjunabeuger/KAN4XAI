import torch
from torch import nn
import torch.nn.functional as F
from actions import get_next_node_indices

"""
All the forward policies for GFlowNet. The `RNNForwardPolicy` contains implementations using vanilla RNN,
GRU, and LSTM. The `CanonicalBackwardPolicy` serves as a placeholder since the backward probabilities is
trivial when the state space has a tree structure.
"""


class RNNForwardPolicy(nn.Module):
    def __init__(self, batch_size, hidden_dim, num_actions,
                 num_layers=1, model='rnn', dropout=0.2, placeholder=-2, one_hot=True, device=None):
        super(RNNForwardPolicy, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.dropout = dropout
        self.placeholder = placeholder
        self.one_hot = one_hot
        self.device = torch.device("mps")
        self.model = model

        # if using one_hot, we turn (sibling, parent) to 2 * num_actions + 2 vector
        # where the additional 2 denotes 2 placeholder symbols
        state_dim = 2 * num_actions + 2 if self.one_hot else 2

        if model == 'rnn':
            self.rnn = nn.RNN(state_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=self.dropout).to(self.device)
        elif model == 'gru':
            self.rnn = nn.GRU(state_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=self.dropout).to(self.device)
        elif model == 'lstm':
            self.rnn = nn.LSTM(state_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=self.dropout).to(self.device)
            self.init_c0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device)
        else:
            raise NotImplementedError("unsupported model: " + model)

        self.fc = nn.Linear(hidden_dim, num_actions).to(self.device)
        self.init_h0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device)

    def actions_to_one_hot(self, siblings, parents):
        # leave the first
        siblings[siblings == self.placeholder] = -1
        parents[parents == self.placeholder] = -1
        sibling_oh = F.one_hot(siblings + 1, num_classes=self.num_actions + 1)
        parent_oh = F.one_hot(parents + 1, num_classes=self.num_actions + 1)
        return torch.cat((sibling_oh, parent_oh), axis=1)

    def forward(self, encodings):
        if encodings[0, 0] == self.placeholder:
            self.h0 = self.init_h0.unsqueeze(1).repeat(1, len(encodings), 1)
            if self.model == 'lstm':
                self.c0 = self.init_c0.unsqueeze(1).repeat(1, len(encodings), 1)

        nodes_to_assign, siblings, parents = get_next_node_indices(encodings, self.placeholder)
        if self.one_hot:
            rnn_input = self.actions_to_one_hot(siblings, parents).to(self.device)
        else:
            rnn_input = torch.stack([siblings, parents], axis=1).to(self.device)

        # match dimension of the hidden state
        rnn_input = rnn_input.unsqueeze(1).float()

        rnn_input = rnn_input.float()
        if self.model == 'lstm':
            output, (self.h0, self.c0) = self.rnn(rnn_input, (self.h0, self.c0))
        else:
            output, self.h0 = self.rnn(rnn_input, self.h0)
        # Get the last output in the sequence
        output = self.fc(output[:, -1, :])
        probabilities = F.softmax(output, dim=1)

        return probabilities.cpu()


class RandomForwardPolicy(nn.Module):
    def __init__(self, num_actions: int):
        super(RandomForwardPolicy, self).__init__()
        self.num_actions = num_actions

    def forward(self, encodings):
        return torch.ones(self.num_actions) / self.num_actions


class CanonicalBackwardPolicy(nn.Module):
    def __init__(self, num_actions: int):
        super(CanonicalBackwardPolicy, self).__init__()
        self.num_actions = num_actions

    def forward(self, encodings: torch.Tensor):
        """
        Calculate the backward probability matrix for a given encoding.
        This downgrades into simply finding the recent action assigned in
        the forward pass due to the tree structure of our env.
        Let (M, T, A) be the (batch size, max tree size, action space dim)
        Args:
            encodings: a (M * T) encoding matrix
        Returns:
            probs: a (M * A) probability matrix
        """
        ecd_mask = (encodings >= 0)
        assert (ecd_mask.sum(axis=1) >= 0).all()
        # get the indices of the recently assigned node using a special trick
        # we want to get the last `True` element of each row, so we multiply the bool
        # with a value that increases with column index, then taking the argmax
        indices = (ecd_mask * torch.arange(1, encodings.shape[1] + 1)).argmax(axis=1)
        actions = encodings[torch.arange(len(encodings)), indices]
        probs = F.one_hot(actions, self.num_actions)
        return probs
