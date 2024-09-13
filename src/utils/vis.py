import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)



def vis_spec(signal,
             fs=44100, nfft=1024, hop=512, 
             fig_width=10, fig_height_per_plot=6, cmap='inferno', vmin=-150, vmax=None,
             use_colorbar=True, tight_layout=False, save_fig='./vis.png'):
    
    # plt.clf()

    # pre-plot
    if len(signal.shape) == 1:
        signal = signal.reshape(1, -1)
    
    num_sig = signal.shape[0]
    fig_size = (fig_width, fig_height_per_plot*num_sig)
    fig, axes = plt.subplots(nrows=num_sig, ncols=1, sharex=True, figsize=fig_size)
    if num_sig == 1:
        axes = [axes]
    
    # iterative plot
    for i, ax in enumerate(axes):
        x = signal[i] + 1e-9
        Pxx, freqs, bins, im = ax.specgram(x, scale='dB', 
                                        Fs=fs, NFFT=nfft, noverlap=hop,
                                        vmin=vmin, vmax=vmax, cmap=cmap)
        if i == num_sig//2:
            ax.set_ylabel('Frequency (Hz)')
        if i == num_sig-1:
            ax.set_xlabel('Time (s)')

    # post-plot
    if use_colorbar:
        plt.colorbar(im, ax=axes, format="%+2.f dB")
    
    if tight_layout:
        plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig)
        plt.close(fig)
        return
    else:
        return fig


if __name__ == '__main__':
    import torchaudio

    filepath = '/home/xbie/Data/dnr_v2/tr/1002/mix.wav'
    x, fs = torchaudio.load(filepath)
    x = x.numpy().reshape(-1)
    
    import librosa
    sr = 44100
    clip = sr * 3
    _, (trim30dBs,_) = librosa.effects.trim(x[:clip], top_db=30)
    print(trim30dBs)
    breakpoint()
    # plt.clf()
    # plt.plot(x)
    # plt.savefig('./tmp.png')
    # plt.close()
    s = 500000
    e = 500000 + 51200
    x_clip = x[:, s:e]
    vis_spec(x_clip)

