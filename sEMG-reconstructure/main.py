import os    #os库是Python标准库，包含几百个函数,常用路径操作、进程管理、环境参数等几类
from torch.autograd import Variable   #torch.autograd是PyTorch的自动差分引擎，可为神经网络训练提供支持。# Variable是torch.autograd中很重要的类。它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息。
import torch
from datetime import datetime   #datetime模块提供表示和处理日期
from torch.optim import lr_scheduler  #torch.optim.lr_scheduler模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。
from torch.utils.tensorboard import SummaryWriter    #`SummaryWriter` 类提供了一个高级 API，用于在给定目录中创建事件文件，并向其中添加摘要和事件。# 该类异步更新文件内容。 这允许训练程序调用方法以直接从训练循环将数据添加到文件中，而不会减慢训练速度。
from torch.utils.tensorboard.summary import hparams     #(超参数)
from utils import loss as Loss
from utils import metrics as Metrics
from data_loader import data_loader
from model import net_tcn
from utils import visulization
from model import net_lstm
from model import net_GRU
from model import net_RNN

import time


# modification for hparams display
class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        logdir = self._get_file_writer().get_logdir()
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="subject&runningGPU")
    parser.add_argument('-s', '--subject', default='7')
    parser.add_argument('-d', '--device', default='0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    subject = args.subject
    input_size = 1
    output_point = 1024
    num_channels = [16, 16, 16, 16]
    kernel_size = 7#2
    dropout = 0.5
    key_size=32
    nheads=4
    batchsize = 96
    windowsize = 1024
    compwindowsize = 512
    stride = 200
    lr = 0.001
    step_size = 100
    gamma = 0.8
    saving_path = 'G:\\result\\output_sEMG-reconstructure 7 0.5\\'+str(subject)+'\\TTCN-'+datetime.now().strftime("%m-%d_%H-%M-%S")
    modol_path = saving_path + '_mod\\'
    fig_path = saving_path + '_fig\\'

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    if not os.path.exists(modol_path):
        os.makedirs(modol_path)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    # #model = net_tcn.TCN(input_size=self.input_dim,
    #                         hidden_size=self.hidden_dim,
    #                         num_layers=self.num_layers,
    #                         dropout=self.dropout,
    #                         batch_first=True,
    #                         bidirectional=False)#.cuda()
    # model = net_RNN.RNN(input_dim=1, output_seq=1024, dropout=dropout, hidden_dim=1, input_seq=512, num_layers=1).cuda()
    # model = net_GRU.GRU(input_dim=1, output_seq=1024, dropout=dropout, hidden_dim=1, input_seq=512, num_layers=1).cuda()
    # model = net_lstm.LSTM(input_dim=1, output_seq=1024, dropout=dropout, hidden_dim=1, input_seq=512, num_layers=1).cuda()    #模型gpu
    model = net_tcn.TCN(input_size=input_size, windowsize=compwindowsize, output_point=output_point, num_channels=num_channels,
                        kernel_size=kernel_size, dropout=dropout,key_size=key_size, nheads=nheads, use_attention=True, en_res=True, visual=True).cuda()    #模型gpu
    tb_writer = SummaryWriter(log_dir=(saving_path), flush_secs=1)
    trainloader = data_loader.load_training(
                                subject=subject,
                                period=[0, 0.80],
                                windowsize=windowsize,
                                stride=stride,
                                batchsize=batchsize,
                                shuffle=True,
                                pinmemory=True,
                                numworkers=0,
                                droplast=True)
    testloader = data_loader.load_testing(
                                subject=subject,
                                period=[0.80, 1],
                                windowsize=windowsize,
                                stride=stride,
                                batchsize=batchsize,
                                pinmemory=True,
                                numworkers=0,
                                droplast=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    bestacc=0

    for epoch in range(500):#1
        model.train()
        collection_loss = 0
        collection_acc = 0
        collection_r2 = 0
        collection_prd = 0
        collection_time = 0

        for step, (orgframe, compframe) in enumerate(trainloader):

            orgframe, compframe = Variable(orgframe.type(torch.FloatTensor)).cuda(), Variable(compframe.type(torch.FloatTensor)).cuda()   #train gpu
            optimizer.zero_grad()

            t0 = time.perf_counter()
            reconsig = model(compframe)
            collection_time += time.perf_counter() - t0

            loss = Loss.mse_loss(orgframe, reconsig).cuda()    #loss gpu
            acc = Metrics.compute_average_sequence_pearsonr(orgframe, reconsig)
            r2 = Metrics.R2_fun(orgframe, reconsig)
            prd = Metrics.matrix_prd(orgframe, reconsig)
            collection_loss += loss
            collection_acc += acc
            collection_r2 += r2
            collection_prd += prd

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('>-', end="")

        print('epoch {:>3} training : loss = {:>4} acc = {:>4} prd = {:>4} r2 = {:>4} timeconsume = {:>4} lr = {:>4}'.format(epoch,
                                                                                        collection_loss/step,
                                                                                        collection_acc[0]/step,
                                                                                        collection_prd/step,
                                                                                        collection_r2[0] / step,
                                                                                        collection_time / step,
                                                                       optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler.step()
        tb_writer.add_scalar('training/loss', collection_loss / step, epoch)
        tb_writer.add_scalar('training/acc', collection_acc[0] / step, epoch)
        tb_writer.add_scalar('training/prd', collection_prd / step, epoch)
        tb_writer.add_scalar('training/r2', collection_r2[0] / step, epoch)
        tb_writer.add_scalar('training/timeconsume', collection_time / step, epoch)


#測試步驟開始
        model.eval()
        collection_loss = 0
        collection_acc = 0
        collection_r2 = 0
        collection_prd = 0
        collection_time = 0
        collection_orgframe = []
        collection_reconsig = []
        collection_compframe = []
        with torch.no_grad():
            for step, (orgframe, compframe) in enumerate(testloader):
                orgframe, compframe = orgframe.type(torch.FloatTensor).cuda(), compframe.type(torch.FloatTensor).cuda()     #test gpu

                t0 = time.perf_counter()
                reconsig = model(compframe)
                collection_time += time.perf_counter() - t0

                loss = Loss.mse_loss(orgframe, reconsig).cuda()
                acc = Metrics.compute_average_sequence_pearsonr(orgframe, reconsig)
                r2 = Metrics.R2_fun(orgframe, reconsig)
                prd = Metrics.matrix_prd(orgframe, reconsig)
                collection_loss += loss
                collection_acc += acc
                collection_r2 += r2
                collection_prd += prd
                collection_orgframe.append(orgframe)
                collection_reconsig.append(reconsig)
                collection_compframe.append(compframe)

        print('epoch {:>3} testing : loss = {:>4} acc = {:>4} prd = {:>4} r2 = {:>4} timeconsume = {:>4} '.format(epoch,
                                                                                 collection_loss/step,
                                                                                 collection_acc[0]/step,
                                                                                 collection_prd/step,
                                                                                 collection_r2[0]/step,
                                                                                 collection_time / step
                                                                                 ))
        tb_writer.add_scalar('testing/loss', collection_loss / step, epoch)
        tb_writer.add_scalar('testing/acc', collection_acc[0] / step, epoch)
        tb_writer.add_scalar('training/prd', collection_prd / step, epoch)
        tb_writer.add_scalar('testing/r2', collection_r2[0] / step, epoch)
        tb_writer.add_scalar('testing/timeconsume', collection_time / step, epoch)
        torch.save({'model': model.state_dict()}, modol_path + 'latest_model.pth')

        if collection_acc[0] / step > bestacc:
            tb_writer.add_hparams({'input_size': str(input_size),
                                   'output_point': str(output_point),
                                   'num_channels': str(num_channels),
                                   'kernel_size': str(kernel_size),
                                   'windowsize': str(windowsize),
                                   'stride': str(stride),
                                   'dropout': str(dropout),
                                   'batchsize': str(batchsize),
                                   'lr': str(lr),
                                   'step_size': str(step_size),
                                   'gamma': str(gamma)
                                   },
                                   {'hparam/acc': collection_acc[0] / step,
                                    'hparam/prd': collection_prd / step,
                                    'hparam/loss': collection_loss / step,
                                    'hparam/r2': collection_r2[0] / step,
                                    'hparam/epoch': epoch})

            collection_orgframe = torch.stack(collection_orgframe).reshape(-1,1,windowsize)
            collection_compframe = torch.stack(collection_compframe).reshape(-1,1,compwindowsize)
            collection_reconsig = torch.stack(collection_reconsig).reshape(-1,1,windowsize)
            rebuilt_orgframe = collection_orgframe[0]
            rebuilt_compframe = collection_compframe[0]
            rebuilt_reconsig = collection_reconsig[0]
            for batchnum in range(1, collection_orgframe.size(0)):
                rebuilt_orgframe = torch.cat((rebuilt_orgframe, collection_orgframe[batchnum][:, -stride:]), dim=1)
                rebuilt_compframe = torch.cat((rebuilt_compframe, collection_compframe[batchnum][:, -int(stride*compwindowsize/windowsize):]), dim=1)
                rebuilt_reconsig = torch.cat((rebuilt_reconsig, collection_reconsig[batchnum][:, -stride:]), dim=1)

            points2show = 100000
            visulization.reconvisualization(rebuilt_orgframe.flatten()[:points2show],
                                            rebuilt_compframe.flatten()[:int(points2show*compwindowsize/windowsize)],
                                            rebuilt_reconsig.flatten()[:points2show], fig_path, epoch)
            torch.save({'rebuilt_orgframe': rebuilt_orgframe, 'rebuilt_compframe': rebuilt_compframe,
                        'rebuilt_reconsig': rebuilt_reconsig}, fig_path + 'final_plot_data.pth')
            torch.save({'model': model.state_dict()}, modol_path + 'best_model'+datetime.now().strftime("%m-%d_%H-%M-%S")+'.pth')
            bestacc = collection_acc[0] / step
    tb_writer.close()


