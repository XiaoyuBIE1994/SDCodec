
import os
import soxr
import torch
import numpy as np
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2

class VisqolMetric():

    def __init__(self,
                 mode='audio',
                 reduction='mean'):

        self.reduction = reduction
        config = visqol_config_pb2.VisqolConfig()
        if mode == "audio":
            self.fs = 48000
            config.audio.sample_rate = self.fs
            config.options.use_speech_scoring = False
            svr_model_path = "libsvm_nu_svr_model.txt"
        elif mode == "speech":
            self.fs = 16000
            config.audio.sample_rate = self.fs
            config.options.use_speech_scoring = True
            svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        else:
            raise ValueError(f"Unrecognized mode: {mode}")
        
        config.options.svr_model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
        
        self.api = visqol_lib_py.VisqolApi()
        self.api.Create(config)

    def __call__(self, est, ref, sr=44100):
        assert est.shape == ref.shape, 'expected same shape, get est: {}, ref: {} instead'.format(est.shape, ref.shape)
        assert len(est.shape) == len(ref.shape) == 3, 'expected BxCxN, get est: {}, ref: {} instead'.format(est.shape, ref.shape)
        B, C, T = est.shape
        est = est.reshape(B*C, -1).detach().cpu().numpy().astype(np.float64)
        ref = ref.reshape(B*C, -1).detach().cpu().numpy().astype(np.float64)

        if sr != self.fs:
            est_list = []
            ref_list = []
            for i in range(est.shape[0]): 
                est_list.append(soxr.resample(est[i], sr, self.fs))
                ref_list.append(soxr.resample(ref[i], sr, self.fs))
            est = np.array(est_list)
            ref = np.array(ref_list)

        ret = []
        for i in range(est.shape[0]):
            ret.append(self.api.Measure(ref[i], est[i]).moslqo)

        if self.reduction == "mean":
            ret = np.mean(ret)
        elif self.reduction == "sum":
            ret = np.sum(ret)
        else:
            ret = ret
        return ret


if __name__ == '__main__':
    eval_visqol = VisqolMetric(mode='audio')

    import soundfile as sf
    audio_name = '001_fragment_10.5'
    ref_audio_path = f'/home/xbie/Data/starnet/starnet_frag_16kHz/{audio_name}.wav'
    target_audio_path = ref_audio_path
    reference, fs_ref = sf.read(ref_audio_path)
    degraded, fs_target = sf.read(target_audio_path)
    print(f'Audio file ref: {fs_ref} Hz, target: {fs_target} Hz')
    
    est = torch.from_numpy(degraded).reshape(1, 1, -1)
    ref = torch.from_numpy(reference).reshape(1, 1, -1)
    metric_visqol = VisqolMetric(mode='audio')
    visqol = metric_visqol(est=est, ref=ref, sr=fs_ref)
    print(f'Visqol: {visqol:.2f}')
