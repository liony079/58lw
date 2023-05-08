import  numpy as np
import math
from PIL import Image
import torch
import scipy.io as scio


# #OMP算法函数
# def cs_omp(y,Phi,K):
#     residual=y  #初始化残差
#     (M,N) = Phi.shape
#     index=np.zeros(N,dtype=int)
#     for i in range(N): #第i列被选中就是1，未选中就是-1
#         index[i]= -1
#     result=np.zeros((N,1))
#     for j in range(K):  #迭代次数
#         product=np.fabs(np.dot(Phi.T,residual))
#         pos=np.argmax(product)  #最大投影系数对应的位置
#         index[pos]=1 #对应的位置取1
#         my=np.linalg.pinv(Phi[:,index>=0]) #最小二乘
#         a=np.dot(my,y) #最小二乘,看参考文献1
#         residual=y-np.dot(Phi[:,index>=0],a)
#     result[index>=0]=a
#     Candidate = np.where(index>=0) #返回所有选中的列
#     return  result, Candidate

# def cs_ompm(y,T_Mat,m):
#     n = len(y)
#     s = 4*math.floor(n/4)
#     hat_x = np.zeros([1,m])
#     Aug_t=[]
#     r_n=y
#     pos_array = np.zeros(s)
#     for times in range(1,s):
#         product=np.abs(np.dot(T_Mat.T,r_n))
#         pos = product.argmax()
#         Aug_t=np.r_[Aug_t,T_Mat[:,pos]]
#         T_Mat[:,pos] = np.zeros([n])
#         aug_x = np.linalg.inv((Aug_t.T*Aug_t))*Aug_t.T*y
#         r_n = y-Aug_t*aug_x
#         pos_array[times]=pos
#     hat_x[pos_array]=aug_x
#     return hat_x

def cs_omp(y,D):
    L=math.floor(3*(y.shape[0])/4)
    residual=y  #初始化残差
    index=np.zeros((L),dtype=int)
    for i in range(L):
        index[i]= -1
    result=np.zeros((1024))
    for j in range(L):  #迭代次数
        product=np.fabs(np.dot(D.T,residual))
        pos=np.argmax(product)  #最大投影系数对应的位置
        index[j]=pos
        my=np.linalg.pinv(D[:,np.where(index>=0)]) #最小二乘,看参考文献1
        a=np.dot(my[:,:,0],y) #最小二乘,看参考文献1
        residual=y-np.dot(D[:,index>=0],a)
    result[index>=0]=a
    return result

if __name__ == "__main__":
    im = scio.loadmat('/home/amax/Documents/sEMG-reconstructure/EMG_sample.mat')['EMG20'].transpose()
    Phi = np.load('/home/amax/Documents/sEMG-reconstructure/Phi.npy')
    waveletbasis = scio.loadmat('/home/amax/Documents/sEMG-reconstructure/coif5wavelets.mat')["ww"]
    transmit = np.dot(Phi, waveletbasis)
    compsig = np.dot(transmit, im.transpose())

    # reconsig = np.random.randn(1,1024)
    # column_rec, Candidate = cs_omp(compsig, Phi, 1000)
    # reverswavlet = np.linalg.inv(waveletbasis)
    # reconsig = np.dot(reverswavlet, column_rec)

    reconsig = cs_omp(compsig,transmit)


    # for i in range(16):
    #     y = np.reshape(compsig[:, i], (512, 1))
    #     column_rec, Candidate = cs_omp(y, Phi, 100)  # 利用OMP算法计算稀疏系数
    #     reconsig[i,:] = np.reshape(column_rec, (1024))
    #     print(np.corrcoef(im[i,:],reconsig[i,:]))


