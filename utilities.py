import numpy as np
from scipy.fftpack import fft,ifft
from scipy import signal
import matplotlib.pyplot as plt

# plot
def PLOT(y,x=None):
    if x is None:
        x = np.arange(0,y.shape[0])
    plt.plot(x,y)
    plt.show()

# stem
def STEM(y,x=None):
    if x is None:
        x = np.arange(0,y.shape[0])
    plt.stem(x,y)
    plt.show()

# fft
def FFT(x):
    xf = abs(fft(x))/len(x)*2
    xf = xf[range(int(len(xf)/2))]
    return xf

# 下采样（平均池化）1D
def downsampling_avg_1d(x):
    return np.mean(np.reshape(x, [int(x.shape[0] / 2), 2]),axis=1)

# 下采样（平均池化）批
def dowmsampling_avg_batch(batch):
    batch2 = np.zeros(shape=[batch.shape[0],3,int(batch.shape[2]/2)],dtype=batch.dtype)
    for idx1 in range(batch2.shape[0]):
        for idx2 in range(3):
            batch2[idx1,idx2,:] = downsampling_avg_1d(batch[idx1,idx2,:])
    return batch2

# 上采样（近邻插值）1D
def upsampling_NN_1d(x):
    return np.reshape(np.tile(x.reshape([-1,1]),(1,2)),[-1])

# 上采样（线性插值）1D
def upsampleing_linear_1d(x,sl):
    # x的长度
    sl0 = len(x)
    # 构建x2
    x2 = np.zeros(shape=sl)
    # sl0 线性分布到 sl
    idxs = [int(np.round(i)) for i in np.linspace(0,sl-1,sl0)]
    for idx,item in zip(idxs,x):
        x2[idx] = item
    # 0处线性插值
    idx = 0
    while idx<sl:
        if x2[idx]==0 and idx!=0:
            s = idx # 记录起点
            a1 = x2[s-1] # 记录上一点
            # 存在连续0 则继续遍历
            while x2[idx]==0:
                idx += 1
            e = idx-1 # 记录终点
            a2 = x2[e+1]
            # 线性插值
            x2[s:e+1] = np.linspace(a1,a2,e-s+1+2)[1:-1]
        idx += 1
    return x2

def upsampling_linear_batch(batch, sl):
    batch2 = np.zeros(shape=[batch.shape[0],sl])
    for idx in range(batch2.shape[0]):
        batch2[idx] = upsampleing_linear_1d(batch[idx],sl)
    return batch2

# 先下采样后上采样
def downsampling_and_upsampling_batch(batch):
    for idx1 in range(batch.shape[0]):
        for idx2 in range(3):
            batch[idx1,idx2,:] = upsampling_NN_1d(downsampling_avg_1d(batch[idx1,idx2,:]))
    return batch

# 数据集频谱图
def batch_fft_mean(batch):
    fft_m = []
    for ss in batch:
        fft = FFT(ss)
        fft_m.append(fft)
    return np.mean(np.array(fft_m), axis=0)
if __name__ == '__main__':
    a = np.arange(0,20)**2
    PLOT(a)
    b = upsampleing_linear_1d(a,50)
    PLOT(b)