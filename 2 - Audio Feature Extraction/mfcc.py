import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct

def freq_to_mel(freq):
    return 2595 * np.log10(1 + freq / 700)

def met_to_freq(mels):
    return 700 * (10 ** (mels / 2595) - 1)

def get_filter_points(fmin, fmax, mel_filter_num, nfft, sampling_rate):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, mel_filter_num + 2)
    freqs = met_to_freq(mels)
    return np.floor((nfft + 1) / sampling_rate * freqs).astype(int)

def create_mfcc(samples, sampling_rate, window_length_ms, overlap, nfft, show_mfcc_plot):
    window_size = int(sampling_rate * window_length_ms / 1000)
    step_size = int(window_size * (1 - overlap))
    start_of_steps = np.arange(0, len(samples), step_size, dtype=int)
    start_of_steps = start_of_steps[start_of_steps + window_size < len(samples)]
    power_spectrum = []
    for start_of_step in start_of_steps:
        step = samples[start_of_step:start_of_step + window_size]
        step *= np.hamming(window_size)
        mag_fft = np.absolute(np.fft.rfft(step, nfft))
        step_power = (mag_fft ** 2) / nfft
        power_spectrum.append(step_power)
    power_spectrum = np.array(power_spectrum)
    freq_min = 0
    freq_high = sampling_rate / 2
    mel_filter_num = 40
    filter_points = get_filter_points(freq_min, freq_high, mel_filter_num, nfft, sampling_rate)
    fbank = np.zeros((mel_filter_num, int(nfft / 2 + 1)))
    for i in range(mel_filter_num):
        f_m_minus = filter_points[i]
        f_m = filter_points[i + 1]
        f_m_plus = filter_points[i + 2]
        for k in range(f_m_minus, f_m):
            fbank[i, k] = (k - filter_points[i]) / (filter_points[i + 1] - filter_points[i])
        for k in range(f_m, f_m_plus):
            fbank[i, k] = (filter_points[i + 2] - k) / (filter_points[i + 2] - filter_points[i + 1])
    filter_banks = np.dot(power_spectrum, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    num_cepstral = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_cepstral + 1)]
    if show_mfcc_plot:
        plt.figure(figsize=(15,5))
        plt.imshow(mfcc.T, aspect='auto', origin='lower')
        plt.show()
    return mfcc

def get_mfcc(audio_path, sampling_rate = None, window_length_ms = 20, overlap = 0.5, nfft = 512, show_mfcc_plot = False):
    samples, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    if len(samples) < sampling_rate:
        samples = np.append(samples, np.ones(sampling_rate - len(samples)) * 1e-6)
    elif len(samples) > sampling_rate:
        samples = samples[:sampling_rate]
    return create_mfcc(samples, sampling_rate, window_length_ms, overlap, nfft, show_mfcc_plot)

def get_noise_augmented_mfcc(audio_path, noise_path, sampling_rate = None, window_length_ms = 20, overlap = 0.5, nfft = 512, show_mfcc_plot = False):
    samples, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    noise_samples, noise_sampling_rate = librosa.load(noise_path, sr=sampling_rate)
    assert noise_sampling_rate == sampling_rate
    noise_beginning = np.random.randint(0, len(noise_samples) - len(samples))
    noise_samples = noise_samples[noise_beginning : (noise_beginning + len(samples))]
    alpha = 0.05
    samples = samples * (1 - alpha) + noise_samples * alpha
    if len(samples) < sampling_rate:
        samples = np.append(samples, np.ones(sampling_rate - len(samples)) * 1e-6)
    elif len(samples) > sampling_rate:
        samples = samples[:sampling_rate]
    return create_mfcc(samples, sampling_rate, window_length_ms, overlap, nfft, show_mfcc_plot)