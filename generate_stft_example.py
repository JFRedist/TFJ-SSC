import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import windows
from args import Path


def spectrogram(x, window, n_overlap, nfft):
    """
    Transform to time-frequency images. This function imitates function spectrogram in Matlab
    Args:
        x (numpy array): Data
        window (int): Size of window function
        n_overlap (int): Number of coincidence points between two segments
        nfft (int): Number of points during Fast Fourier Transform
    """
    len_x = len(x)
    step = window - n_overlap
    nn = nfft // 2 + 1
    num_win = int(np.floor((len_x - n_overlap) / (window - n_overlap)))
    spectrogram_data = []
    # Hamming window default
    win = windows.hamming(window)
    for i in range(num_win):
        subdata = x[i * step: i * step + window]
        F = fft(subdata * win, n=nfft)
        spectrogram_data.append(F[:nn])
    spectrogram_data = np.array(spectrogram_data)
    return spectrogram_data


def load_single_eeg_segment():
    """Load a single 30-second EEG data segment"""
    path = Path()
    channel = 'EEG_Fpz-Cz'  # Using Fpz-Cz channel as example
    
    # Get data file list
    data_dir = os.path.join(path.path_raw_data, channel)
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return None
        
    data_files = sorted(os.listdir(data_dir))
    if not data_files:
        print(f"Data directory is empty: {data_dir}")
        return None
    
    # Load first 30-second segment from first file
    print(f"Loading file: {data_files[0]}")
    data = np.load(os.path.join(data_dir, data_files[0])).astype('float32')
    
    # Return first 30-second segment
    if data.shape[0] > 0:
        return data[0, 0, :]  # 获取第一个30秒片段
    else:
        print("Failed to load data")
        return None


def generate_stft_example():
    """Generate and save 30-second STFT spectrogram example"""
    # Load single 30-second EEG segment
    eeg_data = load_single_eeg_segment()
    if eeg_data is None:
        return
    
    # Set STFT parameters
    fs = 100  # Sampling rate
    win_size = 2  # Window size (seconds)
    overlap = 1  # Overlap size (seconds)
    nfft = 256  # FFT points
    
    # 计算STFT
    print("Calculating STFT...")
    stft_result = spectrogram(eeg_data, win_size * fs, overlap * fs, nfft)
    stft_db = 20 * np.log10(abs(stft_result))
    
    # 绘制时频图
    plt.figure(figsize=(12, 8))
    
    # 计算时间和频率轴
    time_points = np.linspace(0, 30, stft_db.shape[0])
    freq_points = np.linspace(0, fs/2, stft_db.shape[1])
    
    # 绘制热图
    plt.pcolormesh(time_points, freq_points, stft_db.T, cmap='jet', shading='gouraud')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    
    # 设置标题和轴标签
    plt.title('30-second EEG STFT Spectrogram (EEG Fpz-Cz channel)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    
    # 保存图像
    output_dir = 'data/sleepEDF-78/examples'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'stft_example.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Spectrogram saved to: {output_path}")
    
    # 显示图像
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    generate_stft_example()