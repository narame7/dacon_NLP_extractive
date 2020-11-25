import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden_dim =300

        self.rnn1 =nn.LSTM(1050, self.hidden_dim,num_layers=2, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(self.hidden_dim, self.hidden_dim,num_layers=2, bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(30 * 2 * 2, 256)

        self.fc2 = nn.Linear(256, 3)

        self.dropout = nn.Dropout(p=0.01)

        torch.nn.init.xavier_uniform_(self.fc1.weight)



    def forward(self, x):

        # c1 = F.relu(self.maxPool(self.conv1(x)))
        # c2 = F.relu(self.maxPool(self.conv2(x)))
        # c3 = F.relu(self.maxPool(self.conv3(x)))
        # x=torch.cat((c1,c2,c3),dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxPool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))


        x = x.view(-1, 30 *2 * 2 )



        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))


        return out
