import torch
import numpy as np

content = torch.load('C:\\Users\\Administrator\\Desktop\\result\\3output_sEMG-reconstructure 1 0.5\\1\\TTCN-11-17_16-10-11_fig\\final_plot_data.pth')
keys = list(content.keys())
print(keys)
for key in keys:
    print(content[key])

#np.savetxt('my_file.txt', content['rebuilt_orgframe'].numpy())

import scipy.io as io
result1 = np.array(content['rebuilt_orgframe'])
np.savetxt('npresult1.txt',result1)
io.savemat('save1.mat',{'result1':result1})

result2 = np.array(content['rebuilt_compframe'])
np.savetxt('npresult2.txt',result2)
io.savemat('save2.mat',{'result2':result2})

result3 = np.array(content['rebuilt_reconsig'])
np.savetxt('npresult3.txt',result3)
io.savemat('save3.mat',{'result3':result3})