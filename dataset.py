"""
Fs:512
f1:
f2:
f3:
"""
import numpy as np
import utilities
import tfr_tools as tfr

# 定义正弦信号
def SIN(Fs,A,f):
    # 时间序列
    t = np.linspace(0, 1, Fs)
    # 随机指定初始相位
    b = np.random.uniform(-np.pi, np.pi)
    # 正弦信号
    S = A*np.sin(2*np.pi*f*t+b)
    return S.astype(np.float32)

# 定义多通道正弦信号
def Multi_channel_SIN(Fs,A1,A2,f1,f2,a1,a2):
    MS = np.zeros(shape=[3,Fs],dtype=np.float32)
    # 随机指定赋值
    A1 = np.random.uniform(A1-3,A1+3)
    A2 = np.random.uniform(A2-3,A2+3)
    A3 = a1*A1+(1-a1)*A2
    # 随机指定频率
    f1 = np.random.uniform(f1-10,f1+10)
    f2 = np.random.uniform(f1-10,f2+10)
    f3 = a2*f1+(1-a2)*f2
    # 生成正弦波
    MS[0] = SIN(Fs,A1,f1)
    MS[1] = SIN(Fs, A2, f2)
    MS[2] = SIN(Fs, A3, f3)
    print('A1:%.2f'%A1,'A2:%.2f'%A2,'A3:%.2f'%A3)
    print('f1:%.2f' % f1, 'f2:%.2f' % f2, 'f3:%.2f' % f3)
    return MS

def batch_nrom(batch):
    MAX = np.max(batch,keepdims=True)
    MIN = np.min(batch,keepdims=True)
    delta = MAX-MIN
    return (batch-MIN)/delta*2-1

def Max_Merge(x1,x2):
    x3 = np.zeros(shape=len(x1))
    for idx in range(len(x1)):
        x3[idx] = np.maximum(x1[idx],x2[idx])
    return x3

def batch_fft_mean(batch):
    fft_m = []
    for ss in batch:
        fft = utilities.FFT(ss)
        fft_m.append(fft)
    return np.mean(np.array(fft_m), axis=0)


if __name__ == '__main__':

    # 设置参数 F0 = [10,40,80] F1 = [20,60,120] , F3 = [15,50,100]
    #          A0 = [10,5,2.5] A2 = [5,2.5,1.25],A3 = [7.5,3.75,1.825]
    Fs = 512
    f1 = 20
    f2 = 50
    A1 = 10
    A2 = 5
    a1 = 0.5
    a2 = 0.5


    # 产生混频
    MS = Multi_channel_SIN(Fs,A1,A2,f1,f2,a1,a2)
    utilities.PLOT(MS[0])
    utilities.PLOT(MS[1])
    utilities.PLOT(MS[2])
    # utilities.STEM(utilities.FFT(MS[0]))

    # 生产数据集
    dataset = np.zeros(shape=[10000,3,Fs],dtype=np.float32)
    for idx in range(dataset.shape[0]):
        dataset[idx] = Multi_channel_SIN(Fs,A1,A2,f1,f2,a1,a2)

    dataset = batch_nrom(dataset)

    # fft
    # utilities.STEM(utilities.batch_fft_mean(dataset[:,0,:]))

    # tfr
    for i in range(6):
        sl = int(512 / (2 ** i))
        if sl < 512:
            dataset = utilities.dowmsampling_avg_batch(dataset)
        utilities.PLOT(dataset[0,0,:])
        tfr.Saving_All_TFR(r'./TFR/MS_%d' % sl, dataset, np.zeros(shape=dataset.shape[0], dtype=np.uint8), 4)


