import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile

MSP_path = 'current_exp/data/aew/arctic_a0001.pt'
MSP_torch = torch.load(MSP_path) 
MSP = np.asarray(MSP_torch) 



plt.figure(figsize=(20,4), dpi=300)
plt.imshow(MSP, origin='lower', cmap=plt.cm.hot, aspect='auto', interpolation='nearest')
plt.colorbar()
plt.title('mel-spectrogram (SD)', fontsize=14)
plt.xlabel('time index', fontsize=14)
plt.ylabel('frequency bin index', fontsize=14)

plt.show()
