import torch


# def compute_points_pearsonr(batch_x, batch_y, overlap):
#     num_seqs = batch_x.shape[0]
#     pcc_list = []
#     for i in range(num_seqs):
#         x = batch_x[i, :]
#         y = batch_y[i, :]
#         pcc = th_pearsonr(x, y)
#         pcc_list.append(pcc)
#     pcc_batch = torch.FloatTensor(pcc_list).mean()
#     return pcc_batch


# def compute_point_pearsonr(batch_x, batch_y, overlap): #BxCxT
#     # batch_x = batch_x.permute(0, 2, 1) # --> BxTxC
#     # batch_y = batch_y.permute(0, 2, 1) # --> BxTxC
#
#     num_seqs = batch_x.shape[0]
#     num_chs = batch_x.shape[-1]
#
#     if "cuda" in batch_x.type():
#         index = torch.arange(num_chs).cuda().unsqueeze(0)
#     else:
#         index = torch.arange(num_chs).unsqueeze(0)
#
#     pcc_list = []
#
#     for i in range(num_seqs):
#         x = batch_x[i, :]
#         y = batch_y[i, :]
#         cc = th_matrixcorr(x, y)
#         pcc = cc.gather(0, index).mean()
#         pcc_list.append(pcc)
#
#     pcc_batch = torch.stack(pcc_list, dim=0).mean()
#     return pcc_batch

def compute_point_output_pearsonr(batch_x, batch_y): #BxCxT
    # batch_x = batch_x.permute(0, 2, 1) # --> BxTxC
    # batch_y = batch_y.permute(0, 2, 1) # --> BxTxC

    num_bat = batch_x.shape[0]
    num_chs = batch_x.shape[1]
    num_pts = batch_x.shape[2]
    if "cuda" in batch_x.type():
        index = torch.arange(num_chs).cuda().unsqueeze(0)
    else:
        index = torch.arange(num_chs).unsqueeze(0)

    bx = batch_x.permute(1,0,2).contiguous().view(num_chs,-1)
    by = batch_y.permute(1,0,2).contiguous().view(num_chs,-1)

    x = bx.permute(1,0)
    y = by.permute(1,0)
    # pcc_list = []
    # for i in range(num_chs):
    #     cc = th_matrixcorr(x, y)
    #     pcc = cc.gather(0, index).mean()
    #     pcc_list.append(pcc)
    cc = th_matrixcorr(x, y)
    pcc = cc.gather(0, index)
    # return torch.stack(pcc_list,dim=0)
    return torch.squeeze(pcc)

def compute_average_sequence_pearsonr(batch_x, batch_y):
    # batch_x # --> BxCxT
    # batch_y # --> BxCxT

    num_bat = batch_x.shape[0]
    num_chs = batch_x.shape[1]
    num_pts = batch_x.shape[2]
    if "cuda" in batch_x.type():
        index = torch.arange(num_bat).cuda().unsqueeze(0)
    else:
        index = torch.arange(num_bat).unsqueeze(0)

    pcc_list = []
    for i in range(num_chs):
        cc = th_matrixcorr(batch_x[:,i,:].permute(1,0), batch_y[:,i,:].permute(1,0))
        pcc = cc.gather(0, index).mean()
        pcc_list.append(pcc)
    return torch.stack(pcc_list,dim=0)
    # return torch.squeeze(pcc)

def th_pearsonr(x, y):
    """
    mimics scipy.stats.pearsonr
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / (r_den + 1e-8)
    return r_val

def th_corrcoef(x):
    """
    mimics np.corrcoef
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    c = torch.clamp(c, -1.0, 1.0)

    return c

def th_matrixcorr(x, y):
    """
    return a correlation matrix between
    columns of x and columns of y.
    So, if X.size() == (1000,4) and Y.size() == (1000,5),
    then the result will be of size (4,5) with the
    (i,j) value equal to the pearsonr correlation coeff
    between column i in X and column j in Y
    """
    mean_x = torch.mean(x, 0)
    mean_y = torch.mean(y, 0)
    xm = x.sub(mean_x.expand_as(x))
    ym = y.sub(mean_y.expand_as(y))
    r_num = xm.t().mm(ym)
    r_den1 = torch.norm(xm, p=2, dim=0, keepdim=True)
    r_den2 = torch.norm(ym, p=2, dim=0, keepdim=True)

    r_den = r_den1.t().mm(r_den2)
    r_mat = r_num.div(r_den + + 1e-8)
    return r_mat

def R2_fun(output, target):
    # 拟合优度R^2
    num_chs = output.shape[1]
    r2_list = []
    for i in range(num_chs):
        y_mean = torch.mean(target[:,i,:],axis=1).unsqueeze(dim=1).repeat(1,target[:,i,:].size(-1))
        r2_list.append(torch.mean(1 - (torch.sum((output[:,i,:] - target[:,i,:]) ** 2 ,axis=1)) / (torch.sum((target[:,i,:] - y_mean) ** 2,axis=1))))
    return torch.stack(r2_list,dim=0)

def matrix_prd(x, y):
    rmse = torch.sum(torch.square(x-y),dim=2)**0.5
    b = torch.sum(torch.square(x),dim=2)**0.5
    return torch.mean(rmse*100/b)

if __name__ == "__main__":
    x = torch.rand([16, 1, 512])
    y = torch.rand([16, 1, 512])

    print(x.shape)
    print(y.shape)
    import numpy as np

    # ncc = np.corrcoef(x.cpu().numpy().squeeze(),
    #                   y.cpu().numpy().squeeze())
    # print(ncc)
    acc = compute_average_sequence_pearsonr(x, y)
    print(acc)
    r2 = R2_fun(x, y)
    print(r2)
    prd = matrix_prd(x,y)
    print(prd)