import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# if torch.cuda.is_available():
#   Chomp1d=Chomp1d(chomp_size=1).cuda()

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        mask = np.array([[1 if i > j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        if input.is_cuda:
            mask = torch.ByteTensor(mask).cuda(input.get_device())
        else:
            mask = torch.ByteTensor(mask)
        # mask = mask.bool()
        input = input.permute(0, 2, 1)  # input: [N, T, inchannels]
        keys = self.linear_keys(input)  # keys: (N, T, key_size)
        query = self.linear_query(input)  # query: (N, T, key_size)
        values = self.linear_values(input)  # values: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))

        weight_temp = F.softmax(temp / self.sqrt_key_size,
                                dim=1)  # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_vert = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_hori = F.softmax(temp / self.sqrt_key_size, dim=2) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp = (weight_temp_hori + weight_temp_vert)/2
        value_attentioned = torch.bmm(weight_temp, values).permute(0, 2, 1)  # shape: (N, T, value_size)
        return value_attentioned, weight_temp  # value_attentioned: [N, in_channels, T], weight_temp: [N, T, T]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, key_size, nheads, use_attention, en_res=True, visual=True, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.en_res = en_res
        self.visual = visual
        self.use_attention = use_attention
        self.nheads = nheads
        if self.use_attention:
            if self.nheads > 1:
                self.attentions = AttentionBlock(n_inputs, key_size, n_inputs)
                self.attentions = [AttentionBlock(n_inputs, key_size, n_inputs) for _ in range(self.nheads)]
                for i, attention in enumerate(self.attentions):
                    self.add_module('attention_{}'.format(i), attention)
                # self.cat_attentions = AttentionBlock(n_inputs * self.nheads, n_inputs, n_inputs)
                self.linear_cat = nn.Linear(n_inputs * self.nheads, n_inputs)
            else:
                self.attention = AttentionBlock(n_inputs, key_size, n_inputs)



        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):

        if self.use_attention == True:
            en_res_x = None
            if self.nheads > 1:
                # will create some bugs when nheads>1
                x_out = torch.cat([att(x) for att in self.attentions], dim=1)
                out = self.net(self.linear_cat(x_out.transpose(1, 2)).transpose(1, 2))
                m = 1
            else:
                # x = x if self.downsample is None else self.downsample(x)
                out_attn, attn_weight = self.attention(x)
                out = self.net(out_attn)
                weight_x = F.softmax(attn_weight.sum(dim=2), dim=1)
                en_res_x = weight_x.unsqueeze(2).repeat(1, 1, x.size(1)).transpose(1, 2) * x
                en_res_x = en_res_x if self.downsample is None else self.downsample(en_res_x)

            res = x if self.downsample is None else self.downsample(x)

            if self.visual:
                attn_weight_cpu = attn_weight.detach().cpu().numpy()
            else:
                attn_weight_cpu = [0] * 10
            del attn_weight

            if self.en_res:
                return self.relu(out + res + en_res_x), attn_weight_cpu
            else:
                return self.relu(out + res), attn_weight_cpu

        else:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)  # return: [N, emb_size, T]


# if torch.cuda.is_available():
#   TemporalBlock=TemporalBlock().cuda()

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, key_size, nheads, use_attention, en_res=True, visual=True, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.use_attention = use_attention
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, key_size=key_size, nheads=nheads,
                                     use_attention=use_attention, en_res=en_res, visual=visual, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        attn_weight_list = []
        if self.use_attention:
            out = x
            for i in range(len(self.network)):
                out, attn_weight = self.network[i](out)
                # print("the len of attn_weight", len(attn_weight))
                # if len(attn_weight) == 64:
                #     attn_weight_list.append([attn_weight[18], attn_weight[19]])
                attn_weight_list.append([attn_weight[0], attn_weight[-1]])
            return out, attn_weight_list
        else:
            return self.network(x)
# if torch.cuda.is_available():
#   TemporalConvNet=TemporalConvNet().cuda()

class TCN(nn.Module):
    def __init__(self, input_size, windowsize, output_point, num_channels, kernel_size, dropout, key_size, nheads, use_attention=True, en_res=True, visual=True):
        super(TCN, self).__init__()
        self.use_attention = use_attention
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, key_size=key_size, nheads=nheads,
                                     use_attention=use_attention, en_res=en_res, visual=visual, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)
        self.linear2 = nn.Linear(windowsize, output_point)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.1)

    def forward(self, x):
        y1 = x
        if self.use_attention:
            y1, attn_weight_list = self.tcn(x)
        else:
            y1 = self.tcn(x)
        y2 = self.linear(y1.transpose(1, 2)).transpose(2, 1)
        return self.linear2(y2)
# if torch.cuda.is_available():
#   TCN=TCN().cuda()


if __name__ == "__main__":
    input = torch.randn([16, 1, 512])
    model = TCN(input_size=1, windowsize=512, output_point=1024, num_channels=[16, 16, 16, 16], kernel_size=3, dropout=0.5)
    print(model)
    print(model(input).shape)
    # plt.scatter(feat[:,0], feat[:,1], c=lab.squeeze())
    # plt.show()