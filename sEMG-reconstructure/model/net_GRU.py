import torch
import torch.nn as nn
import torch.nn.functional as F
class GRU(nn.Module):
    def __init__(self, input_dim, input_seq, output_seq, dropout, hidden_dim, num_layers):
        super(GRU, self).__init__()

        self.input_dim = input_dim
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True,
                            bidirectional=False)
        self.fc = nn.Sequential(nn.Linear(self.input_seq, self.output_seq))
        # self.classifier = nn.Softmax(dim=1)
        # self.fc = nn.Sequential(nn.Linear(self.hidden_dim*2, self.output_dim),
        #                         nn.ReLU(inplace=True))
        # self.fc2 = nn.Sequential(nn.Linear(self.time_step, self.out_point),
        #                         nn.ReLU(inplace=True))
        # if iscuda:
        #     self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
        #             torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())
        # else:
        #     self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
        #             torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        # self._init_weight()
        # self.apply(self.weight_init)
    # def _init_weight(self):
    #     for m in self.modules():
    #       m.weight = nn.init.xavier_uniform_(m.weight)
            # if isinstance(m, nn.Linear):
            #     nn.init.xavier_normal_(m.weight)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
    # def weight_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm1d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)


    def forward(self, input):
        # input = F.normalize(input, p=2, dim=2)# BxTxC
        input = input.permute(0, 2, 1).contiguous()
        lstm_out, hidden = self.gru(input)
        output = self.fc(lstm_out.squeeze())
        return output.unsqueeze(1)