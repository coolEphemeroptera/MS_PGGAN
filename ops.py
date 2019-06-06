import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# 获取张量shape
def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

# 获取归一化权值（equalized learning rate）
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    """
    HE公式：0.5*n*var(w)=1 , so：std(w)=sqrt(2)/sqrt(n)
    """
    # 某卷积核参数个数或dense层输入节点数目
    # conv_w:[w,fmaps1,fmaps2] or [h,w,fmaps1,fmaps2]
    # mlp_w:[fmaps1,fmaps2]
    if fan_in is None: fan_in = np.prod(shape[:-1])
    # He init
    std = gain / np.sqrt(fan_in)
    # 归一化
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return  tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal())*wscale
    else:
        return  tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0,std))

# 定义像素归一化操作（pixel normalization）
def PN(x):
    if len(x.shape) > 2:
        axis_ = 3
    else:
        axis_ = 1
    epsilon = 1e-8
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis_, keepdims=True) + epsilon)

# 1d卷积
def conv1d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    # x:[N,sl,fmaps1]
    # w:[w,fmaps1,fmaps2]
    w = get_weight([ kernel, x.shape[2].value, fmaps],gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    """
    tf.nn.conv1d:
                input:[N,sl,fmaps1]
                filter:[W,fmaps1,fmaps2]
                output:[N,sl,fmaps2]
    """
    return tf.nn.conv1d(x, w, stride=1, padding='SAME', data_format='NWC')

def conv2d(x, fmaps, k_h,k_w, gain=np.sqrt(2), use_wscale=False):
    # assert kernel >= 1 and kernel % 2 == 1
    # x:[N,elects,sl,fmaps1]
    # w:[h,w,fmaps1,fmaps2]
    w = get_weight([ k_h,k_w, x.shape[3].value, fmaps],gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    """
    tf.nn.conv2d:
                input:[N,elects,sl,fmaps1]
                filter:[h,w,fmaps1,fmaps2]
                output:[N,elects,sl,fmaps2]
    """
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')

# 添加偏置
def add_bias(x):
    # nums(b) = channels
    if len(x.shape) == 2:#[N,fmaps]
        b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros(), dtype=x.dtype)
        return x + b # for FC
    else:# [N,elects,sl,fmaps]
        b = tf.get_variable('bias', shape=[x.shape[3]], initializer=tf.initializers.zeros(), dtype=x.dtype)
        return x + tf.reshape(b, [1, 1,1, -1]) # for CONV

# dense
def dense(x, fmaps,gain=np.sqrt(2), use_wscale=False):
    # 平铺至1D [N,flat(fmaps1)]
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    # 获取权值
    w = get_weight([x.shape[1].value, fmaps],gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    """
    tf.matmul:
             input:[N,fmaps1]
             w: [fmaps1,fmaps2]
             output: [N,fmaps2]
    """
    return tf.matmul(x, w)

# leaky relu
def lrelu(x):
    return tf.nn.leaky_relu(x,alpha=0.2,name='lrelu')

# 上采样[N,elects,sl,fmaps]
def upscale(x):
    _,elects,sl,fmaps = int_shape(x)
    N = tf.shape(x)[0]
    us = []
    for idx in range(sl):
        slice = tf.slice(x,[0,0,idx,0],[N,3,1,fmaps])
        us.append(slice)
        us.append(slice)
    return tf.concat(us,axis=2)

# 下采样
def downscale(x):
    return tf.layers.average_pooling2d(x,[1,2],[1,2],padding='valid')

# 添加多样性特征
def MinibatchstateConcat(input, averaging='all'):
    # input:[N,H,W,fmaps]
    s = input.shape
    # 获取批大小
    group_size = tf.shape(input)[0]
    """
    计算方法：
            (1)先计算N个特征图的标准差得到特征图fmap1:[1,H,W,fmaps]
            (2)对fmap1求均值 得到值M1:[1,1,1,1]
            (3)复制扩张M2得到N个特征图fmap2:[N,H,W,1]
            (4)将fmap2添加至每个样本的特征图中
    """
    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print ("nothing")
    vals = tf.tile(vals, multiples=(group_size, s[1].value, s[2].value, 1))
    return tf.concat([input, vals], axis=3)

 # 定义特征图数目
def nf(scale):
    NF = [50,50,50,50,50,50]
    return NF[scale]

# 定义信号长度
def sl(scale):
    SL = [16,32,64,128,256,512]
    return SL[scale]

# 定义生成器卷积块
def G_CONV_BLOCK(x, scale,use_wscale=False):
    # 上采样
    with tf.variable_scope('upscale'):
        x = upscale(x)
    # CONV0
    with tf.variable_scope('CONV0'):
        x = PN(lrelu(add_bias(conv2d(x,fmaps=nf(scale), k_h=3, k_w=9, use_wscale=use_wscale))))
    # CONV1
    with tf.variable_scope('CONV1'):
        x = PN(lrelu(add_bias(conv2d(x, fmaps=nf(scale), k_h=3, k_w=9, use_wscale=use_wscale))))
    return x

# 定义判别器卷积块
def D_CONV_BLOCK(x, scale):
    # CONV0
    with tf.variable_scope('CONV0'):
        x = lrelu(add_bias(conv2d(x, fmaps=nf(scale),k_h=3, k_w=9, use_wscale=True)))
    # CONV1,增加特征图个数,fmaps数量改变发生在该卷积，即nf(level) to nf(level-1)
    with tf.variable_scope('CONV1'):
        x = lrelu(add_bias(conv2d(x, fmaps=nf(scale-1), k_h=3,k_w=9, use_wscale=True)))
    # 下采样
    with tf.variable_scope('dowbscale'):
        x = downscale(x)
    return x

# 定义toSS
def toSS(x,level,use_wscale):
    with tf.variable_scope('level_%d_toSS' % level):
        return add_bias(conv2d(x, fmaps=1,k_h=1, k_w=1, gain=1, use_wscale=use_wscale))

# 定义fromSS
def fromSS(x,level,fmaps,use_wscale):
    with tf.variable_scope('level_%d_fromSS' % (level)):
        return lrelu(add_bias(conv2d(x, fmaps=fmaps, k_h=1, k_w=1,use_wscale=use_wscale)))

"""
----------------------------------------------- 统计参数 ---------------------------------------------------------
"""
 # counting total to vars
def COUNT_VARS(vars):
    total_para = 0
    for variable in vars:
        # get each shape of vars
        shape = variable.get_shape()
        variable_para = 1
        for dim in shape:
            variable_para *= dim.value
        total_para += variable_para
    return total_para

# display paras infomation
def ShowParasList(d_vars,g_vars,level,isTrans):
    p = open('./structure/level%d_isTrans_%s_Paras.txt'%(level,isTrans), 'w')
    # D paras
    print('正在记录Discriminator参数信息..')
    p.writelines(['Discriminator_vars_total: %d\n'%COUNT_VARS(d_vars)])
    for variable in d_vars:
        p.writelines([variable.name, str(variable.get_shape()),'\n'])

    p.writelines(['\n','\n','\n'])
    # G paras
    print('正在记录Generator参数信息..')
    p.writelines(['Generator_vars_total: %d\n' % COUNT_VARS(d_vars)])
    for variable in g_vars:
        p.writelines([variable.name, str(variable.get_shape()), '\n'])
    p.close()

 # 定义参数匹配检查，用来检查当前网络的部分参数是否可以使用上一级网络的参数
def VARS_MATCH(old_model_path, vars):
    # 获取模型文件名
    ckpt = tf.train.get_checkpoint_state(old_model_path)
    latest = ckpt.model_checkpoint_path
    # 读取模型
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(latest)
    # 获取所有变量
    var_to_shape_map = reader.get_variable_to_shape_map()
    # 检查型号是否匹配
    for key in var_to_shape_map.keys():
        tensorName = key
        tensorShape = var_to_shape_map[key]
        for var in vars:
            if tensorName in var.name:
                assert list(var.get_shape()) == tensorShape

# build related dirs
def GEN_DIR():
    if not os.path.isdir('ckpt'):
        print('DIR:ckpt NOT FOUND，BUILDING ON CURRENT PATH..')
        os.mkdir('ckpt')
    if not os.path.isdir('trainLog'):
        print('DIR:ckpt NOT FOUND，BUILDING ON CURRENT PATH..')
        os.mkdir('trainLog')
    if not os.path.isdir('structure'):
        print('DIR:ckpt NOT FOUND，BUILDING ON CURRENT PATH..')
        os.mkdir('structure')

# 保存训练记录
def Saving_Train_Log(filename,var,dir=r'./trainLog'):
    var = np.array(var)
    f = open(os.path.join(dir,filename),'wb')
    pickle.dump(var,f)
    f.close()
    print('成功保存记录：%s!'%filename)

# 打印训练信息
def print_traing_info(steps,max_iters,level,sl,isTrans,train_loss_d,train_loss_g,Wasserstein,SLOPES):
    print('level:%d(sl = %d)..' % (level, sl),
          'isTrans:%s..' % isTrans,
          'step:%d/%d..' % (steps, max_iters),
          'Discriminator Loss: %.4f..' % (train_loss_d),
          'Generator Loss: %.4f..' % (train_loss_g),
          'Wasserstein dist:%.4f..' % Wasserstein,
          'Slopes:%.4f..' % SLOPES)

# 在线绘图
def runtime_showing(steps,isTrans,sl,axes,x1,x2,x3,y1,y2,y3):
    # 显示real信号
    axes[0][0].cla()
    axes[0][0].plot(x1)
    axes[0][0].set_title('SS_real0____steps:%d____isTrans:%s___sl:%d' % (steps,isTrans,sl))
    axes[1][0].cla()
    axes[1][0].plot(x2)
    axes[1][0].set_title('SS_real1____steps:%d____isTrans:%s___sl:%d' % (steps,isTrans,sl))
    axes[2][0].cla()
    axes[2][0].plot(x3)
    axes[2][0].set_title('SS_real2____steps:%d____isTrans:%s___sl:%d' % (steps, isTrans, sl))
    # 显示样本信号
    axes[0][1].cla()
    axes[0][1].plot(y1)
    axes[0][1].set_title('SS_fake0____steps:%d____isTrans:%s___sl:%d' % (steps, isTrans, sl))
    axes[1][1].cla()
    axes[1][1].plot(y2)
    axes[1][1].set_title('SS_fake1____steps:%d____isTrans:%s___sl:%d' % (steps, isTrans, sl))
    axes[2][1].cla()
    axes[2][1].plot(y3)
    axes[2][1].set_title('SS_fake2____steps:%d____isTrans:%s___sl:%d' % (steps, isTrans, sl))
    plt.tight_layout(h_pad=3)
    plt.show()
    plt.pause(1 / 1000)