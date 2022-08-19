import numpy as np
import torch
from nnmnkwii.preprocessing.alignment import DTWAligner
from utils.audio_processing import layers


class Transform_Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X):
        for t in self.transforms:
            X = t(X)
        return X

class Pair_Transform_Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X, Y):
        for t in self.transforms:
            X, Y = t(X, Y)
        return X, Y

class apply_MVN(object):
    def __init__(self, X_mean, X_std):
        self.X_mean = X_mean
        self.X_std = X_std
    def __call__(self, X):
        X_norm = (X - self.X_mean)/self.X_std
        return X_norm


class apply_delta_deltadelta(object):
    # Adopted from nnmnkwii source code https://github.com/r9y9/nnmnkwii
    
    def delta(self, x, window):

        T, D = x.shape
        y = np.zeros_like(x)
        for d in range(D):
            y[:, d] = np.correlate(x[:, d], window, mode = "same")
        return y
    
    def apply_delta_windows(self, x, windows):

        T, D = x.shape
        assert len(windows) > 0
        combined_features = np.empty((T, D * len(windows)), dtype=x.dtype)
        for idx, (_, _, window) in enumerate(windows):
            combined_features[:, D * idx:D * idx + D] = self.delta(x, window)
        return combined_features
    
    def __call__(self, ema):
    
        windows = [(0, 0, np.array([1.0])), 
                   (1, 1, np.array([-0.5, 0.0, 0.5])),
                   (1, 1, np.array([1.0, -2.0, 1.0]))]
        
        ema_delta = self.apply_delta_windows(ema.numpy(), windows)
        
        return torch.FloatTensor(ema_delta)


class apply_delta_deltadelta_Src(apply_delta_deltadelta):
    def __call__(self, src_wav, tar_wav):        
        return super().__call__(src_wav), src_wav

class apply_delta_deltadelta_Tar(apply_delta_deltadelta):
    def __call__(self, src_wav, tar_wav):        
        return src_wav, super().__call__(src_wav)

class apply_delta_deltadelta_Src_Tar(apply_delta_deltadelta):
    def __call__(self, src_wav, tar_wav):        
        return super().__call__(src_wav), super().__call__(src_wav)

############### Apply dynamic time warping (DTW) on the src-tar pairs ############

class apply_DTW(object):
    
    def DTW_alignment(self, src, tar):
        from fastdtw import fastdtw
        import numpy as np
        dist, path = fastdtw(src, tar)
        src, tar= src.unsqueeze(0), tar.unsqueeze(0)
        src_align, tar_align = DTWAligner().transform((src.numpy(), tar.numpy()))

        return torch.FloatTensor(src_align.squeeze(0)), torch.FloatTensor(tar_align.squeeze(0))
    
    def __call__(self, src, tar):
    
        src_align, tar_align = self.DTW_alignment(src, tar)
        
        return src_align, tar_align


############### Audio transformation #######################
class change_wav_sampling_rate(object):
    def __init__(self, org_fs = 16000, tar_fs=22050):
        self.tar_fs = tar_fs
        self.org_fs = org_fs
    
    def __call__(self, wav):
        import librosa
        y_out = np.expand_dims(librosa.resample(wav.astype(float), self.org_fs, self.tar_fs), axis = 1)
        return y_out

class pair_change_wav_sampling_rate(object):
    def __init__(self, org_fs = 16000, new_fs=22050):
        self.tar_fs = tar_fs
        self.org_fs = org_fs
    
    def __call__(self, src_wav, tar_wav):
        import librosa
        src_out = np.expand_dims(librosa.resample(src_wav[:,0], self.org_fs, self.new_fs), axis = 1)
        tar_out = np.expand_dims(librosa.resample(tar_wav[:,0], self.org_fs, self.new_fs), axis = 1)
        return y_out

class wav2melspec(object):
    def __init__(self, sampling_rate=22050, filter_length=1024, hop_length=256, win_length=1024, 
                 n_mel_channels=80, mel_fmin=0.0, mel_fmax=8000.0):

        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin, self.mel_fmax = mel_fmin, mel_fmax
        self.stft = layers.TacotronSTFT(self.filter_length, self.hop_length, self.win_length,
                    self.n_mel_channels, self.sampling_rate, self.mel_fmin, self.mel_fmax)
    
    def __call__(self, wav):
        audio = torch.FloatTensor(wav.astype(np.float32))
     #   audio_norm = audio / max_wav_value
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec.T

class pair_wav2melspec(wav2melspec):
    def __call__(self, src_wav, tar_wav):        
        return super().__call__(src_wav), super().__call__(tar_wav)

############### ATS transformation #######################


class apply_delta_deltadelta_EMA_ATS(apply_delta_deltadelta):
    def __call__(self, ema, wav):        
        return super().__call__(ema), wav


class change_wav_sampling_rate_ATS(change_wav_sampling_rate):  
    def __call__(self, ema, wav):        
        return ema, super().__call__(wav)

class wav2melspec_ATS(wav2melspec):
    def __call__(self, ema, wav):        
        return ema, super().__call__(wav)

class ema_wav_length_match(object):
    '''
    scale ema according to wav
    '''
    def __call__(self, ema, wav):
        from scipy import ndimage
        scale_ratio = wav.shape[0] / ema.shape[0]
        ema_align = np.empty([wav.shape[0], ema.shape[1]])
        for i in range(ema.shape[1]):
            ema_align[:,i] = ndimage.zoom(ema[:,i], scale_ratio)
        return ema_align, wav

class padding_end(object):
    def __init__(self, max_len = 240):
        self.max_len = max_len
    
    def __call__(self, ema, wav):
        ema_tensor = torch.tensor(ema)
        pad_len = self.max_len - ema_tensor.shape[0]
        ema_pad_row, wav_pad_row = ema_tensor[-1,:], wav[-1,:]
        ema_pad, wav_pad = ema_pad_row.expand(pad_len, -1), wav_pad_row.expand(pad_len, -1)
        ema_padded, wav_padded = torch.cat((ema_tensor, ema_pad), dim = 0), torch.cat((wav, wav_pad), dim = 0)
        return ema_padded, wav_padded

class zero_padding_end(object):
    def __init__(self, max_len = 240):
        self.max_len = max_len
    
    def __call__(self, ema, wav):
        ema_tensor = torch.tensor(ema)
        pad_len = self.max_len - ema_tensor.shape[0]
        ema_pad_row, wav_pad_row = torch.zeros(ema.shape[1]), torch.zeros(wav.shape[1])
        ema_pad, wav_pad = ema_pad_row.expand(pad_len, -1), wav_pad_row.expand(pad_len, -1)
        ema_padded, wav_padded = torch.cat((ema_tensor, ema_pad), dim = 0), torch.cat((wav, wav_pad), dim = 0)
        return ema_padded, wav_padded

class apply_src_MVN(object):
    def __init__(self, X_mean, X_std):
        self.X_mean = X_mean
        self.X_std = X_std
    def __call__(self, X, Y):
        X_norm = (X - self.X_mean)/self.X_std
        return X_norm, Y

class apply_tar_MVN(object):
    def __init__(self, Y_mean, Y_std):
        self.Y_mean = Y_mean
        self.Y_std = Y_std
    def __call__(self, X, Y):
        Y_norm = (Y - self.Y_mean)/self.Y_std
        return X, Y_norm
