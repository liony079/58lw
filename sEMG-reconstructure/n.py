import numpy as np
windowsize = 1024
# Phi = np.random.randn(int(windowsize * 0.3), windowsize)
Phi = np.random.randn(int(windowsize * 0.7), windowsize)
np.save('Phi 0.7.npy',Phi)