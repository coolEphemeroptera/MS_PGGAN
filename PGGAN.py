import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import tfr_tools as tfr
import utilities
from ops import *
"""
信号tf格式为：[N,elects,sl,fmaps]
"""

#************************************* 定义生成器和判别器 *************************************************************#
# 定义生成器
def Generator_PG(latents,level,reuse = False,isTransit = False,trans_alpha = 0.0):
    """
    说明：（1）Generator构成：scale_0 + scale_1~level + toEEG , 其中toEEG层将全部特征图合成EEG
          (2) 过渡阶段：① 本阶段EEG将融合上一阶段EEG输出。对于上一阶段EEG处理层而言，通过特征图上采样匹配大小，再toEEG再融合。
                      ② 上一阶段toEEG的卷积核参数对于上采样后的特征图依然有效
    """
    # ***************** 构造渐增架构 ******************************#
    with tf.variable_scope('generator', reuse=reuse):
        # (1)构造初始架构 (scale = 0)
        with tf.variable_scope('scale_%d'%(0)):
            with tf.variable_scope('Dense0' ):
                x = dense(latents,fmaps=3*sl(0)*nf(0),gain=np.sqrt(2)/4,use_wscale=True)
                x = tf.reshape(x,[-1,3,sl(0),nf(0)])
                x =lrelu(add_bias(x))

        # (2)构造高级架构（scale = 1~level）
        for scale in range(1, level + 1):
            # 过渡阶段：在建立最后网络之前，获取当前SS并上采样
            if scale == level and isTransit:
               SS0 = upscale(x)  # 上采样
               SS0 = toSS(SS0,scale-1,use_wscale=True)# toEEG
            with tf.variable_scope('scale_%d'%scale):
                x = G_CONV_BLOCK(x,scale,use_wscale=True)# 卷积层拓展

        # (3)toSS
        SS1 = toSS(x,level,use_wscale=True)
        if isTransit:
            return trans_alpha*SS1+(1-trans_alpha)*SS0
        else:
            return SS1

# 定义判别器
def Discriminator_PG(SS,level,reuse = False,isTransit = False,trans_alpha = 0.0):

    """
    说明：（1）Discriminator构成：fromEEG + scale_level~1 + scale_0 ,其中fromEEG是对EEG信号分解
         （2）过渡阶段：①本阶段新的网络层scale_level 分解的结果融合上一阶段fromEEG。上一阶段通过下采样匹配大小，再fromEEG再融合
                      ②上一阶段fromEEG卷积核参数对于本阶段下采样后的依然EEG有效
    """
    # ***************** 构造渐增架构 ******************************#
    with tf.variable_scope("discriminator", reuse=reuse):
        # (1) fromSS
        # 降采样分解
        if isTransit:
            SS0 = downscale(SS)  # 0.5x
            x0 = fromSS(SS0, level - 1, nf(level - 1),use_wscale=True)  # fromSS
        # 新增网络层分解
        x = fromSS(SS, level, nf(level),use_wscale=True)

        # (2) 构造高级架构(level~1)
        for scale in range(level,0,-1):
            with tf.variable_scope('scale_%d' % (scale)):
                x = D_CONV_BLOCK(x,scale) # 拓展卷积层
                # 在新建第一层卷积层后，获取该层的卷积结果x。在过渡阶段实现过渡
                if scale==level and isTransit:
                    x = trans_alpha*x+(1-trans_alpha)*x0

        # (3) 构造终极架构(level = 0)
        with tf.variable_scope('scale_%d' % (0)):
            # 加入多样性特征
            x = MinibatchstateConcat(x)
            with tf.variable_scope('Dense0'):
                x = lrelu(add_bias(dense(x, fmaps=1, use_wscale=True)))
    return  x
#****************************************** 构建PGGAN计算图 **********************************************************#
def PGGAN(
        latents_size, # 噪声型号
        batch_size, # 批型号
        lowest,# 最低网络级数
        highest,#最高网络级数
        level,# 目标网络等级
        isTransit, # 是否过渡
        data_size,  # 数据集数量
        epochs # 迭代轮回次数
        ):
    #-------------------- 超参 --------------------------#
    SL = [16, 32, 64, 128, 256, 512]
    learning_rate = 0.001
    lam_gp = 10
    lam_eps = 0.001
    beta1 = 0.0
    beta2 = 0.99
    max_iters = int(epochs * data_size / batch_size)
    n_critic = 5  # 判别器训练次数

    # ---------- （1）创建目录和指定模型路径 -------------#
    GEN_DIR()
    # 当前模型路径
    model_path = './ckpt/PG_level_%d_isTrans_%s' % (level, isTransit)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)  # 创建目录
    # 上一级网络模型路径
    if isTransit:
        old_model_path = r'./ckpt/PG_level_%d_isTrans_%s/' % (level - 1, not isTransit)  # 上一阶段稳定模型
    else:
        old_model_path = r'./ckpt/PG_level_%d_isTrans_%s/' % (level, not isTransit)  # 该阶段过度模型

    # --------------------- (2)定义输入输出 --------------#
    # 信号长度
    sl = SL[level]
    # 定义噪声输入
    latents = tf.placeholder(name='latents',shape=[None,latents_size],dtype=tf.float32)
    # 定义数据输入
    real_SS = tf.placeholder(name='real_images',shape=[None,3,sl,1],dtype=tf.float32)
    # 训练步数
    train_steps = tf.Variable(0,trainable=False,name='train_steps',dtype=tf.float32)

    # 生成器和判别器输出
    fake_SS = Generator_PG(latents,level,reuse = False,isTransit = isTransit,trans_alpha = train_steps/max_iters)
    d_real_logits = Discriminator_PG(real_SS,level,reuse = False,isTransit = isTransit,trans_alpha = train_steps/max_iters)
    d_fake_logits = Discriminator_PG(fake_SS,level,reuse = True,isTransit = isTransit,trans_alpha = train_steps/max_iters)

    # ------------ (3)Wasserstein距离和损失函数 --------------#
    # 定义wasserstein距离
    Wass = tf.reduce_mean(d_real_logits - d_fake_logits)

    # 定义G,D损失函数
    d_loss = -Wass  # 判别器损失函数
    g_loss = -tf.reduce_mean(d_fake_logits)  # 生成器损失函数

    # 基于‘WGAN-GP’的梯度惩罚
    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)  # 获取[0,1]之间正态分布
    alpha = alpha_dist.sample((batch_size, 1,1,1))
    interpolated = real_SS + alpha * (fake_SS - real_SS)  # 对真实样本和生成样本之间插值
    inte_logit = Discriminator_PG(interpolated, level, reuse=True, isTransit=isTransit,
                                  trans_alpha=train_steps / max_iters)  # 求得对应判别器输出

    # 求得判别器梯度
    gradients = tf.gradients(inte_logit, [interpolated, ])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2,3]))
    slopes_m = tf.reduce_mean(slopes)

    # 定义惩罚项
    # gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
    gradient_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0,slopes-1)))

    # d_loss加入惩罚项
    d_loss += gradient_penalty * lam_gp * tf.maximum(0.0,Wass)
    # d_loss += gradient_penalty * lam_gp

    # 零点偏移修正
    # d_loss += tf.reduce_mean(tf.square(d_real_logits)) * lam_eps

    # ------------ (4)模型可训练参数提取 --------------#
    # 获取G,D 所有可训练参数
    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
    ShowParasList(d_vars,g_vars,level,isTransit)

    # 提取本阶段各级网络层参数（不含EEG处理层）
    d_vars_c = [var for var in d_vars if 'fromSS' not in var.name] # discriminator/scale_(0~level)/
    g_vars_c = [var for var in g_vars if 'toSS' not in var.name] # generator/scale_(0~level)/

    # 提取上一阶段各级网络层参数（不含EEG处理层）
    d_vars_old = [var for var in d_vars_c if 'scale_%d' % level not in var.name] # discriminator/scale_(0~level-1)/
    g_vars_old = [var for var in g_vars_c if 'scale_%d' % level not in var.name] # generator/scale_(0~level-1)/

    # 提取本阶段所有EEG处理层参数
    d_vars_ss = [var for var in d_vars if 'fromSS' in var.name] # discriminator/level_*_fromEEG/
    g_vars_ss = [var for var in g_vars if 'toSS'  in var.name] # generator/level_*_toEEG/

    # 提取上一阶段EEG处理层参数
    d_vars_ss_old = [var for var in d_vars_ss if 'level_%d_fromSS' % level not in var.name] # discriminator/level_level-1_fromEEG/
    g_vars_ss_old = [var for var in g_vars_ss if 'level_%d_toSS' % level not in var.name] # generator/level_level-1_fromEEG/

    # 提取上一阶段全部变量
    old_vars = d_vars_old+g_vars_old+d_vars_ss_old+g_vars_ss_old

    # ------------ (5)梯度下降 --------------#
    # G,D梯度下降方式
    d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         beta1=beta1,
                                         beta2=beta2).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         beta1=beta1,
                                         beta2=beta2).minimize(g_loss, var_list=g_vars, global_step=train_steps)
    # 为保持全局平稳学习，我们将保存参数的更新状态
    all_vars = tf.all_variables()
    adam_vars = [var for var in all_vars if 'Adam' in var.name]
    adam_vars_old = [var for var in adam_vars if 'level_%d' % level not in var.name and 'scale_%d' % level not in var.name]

    # ------------ (6)模型保存与恢复 ------------------#
    # 保存本阶段所有变量
    saver = tf.train.Saver(d_vars + g_vars + adam_vars,max_to_keep=3)
    # 提取上一阶段所有变量
    if level > lowest:
        VARS_MATCH(old_model_path, old_vars)  # 核对
        old_saver = tf.train.Saver(old_vars + adam_vars_old)

    # ------------ (7)数据集读取（TFR） --------------#
    # read TFR
    [num,data, label] = tfr.Reading_TFR(sameName=r'./TFR/MS_%d-*'%(sl), isShuffle=False, datatype=tf.float32,
                                        labeltype=tf.uint8)
    # # get batch
    [num_batch,data_batch, label_batch] = tfr.Reading_Batch_TFR( num,data, label, data_size=sl*3,
                                                                  label_size=1, isShuffle=False,batchSize=batch_size)

    # ------------------ (8)迭代 ---------------------#
    # GPU配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 保存文件
    losses = []
    GenLog = []
    WASS = []

    # 开启会话
    with tf.Session(config=config) as sess:

        # 全局和局部变量初始化
        init = (tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init)

        # 开启协调器
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 恢复历史架构
        if level>lowest:
            if isTransit: #如果处于过渡阶段
                old_saver.restore(sess,tf.train.latest_checkpoint(old_model_path))# 恢复历史模型
            else:# 如果处于稳定阶段
                saver.restore(sess,tf.train.latest_checkpoint(old_model_path))# 继续训练该架构

        # 迭代
        time_start = time.time()  # 开始计时
        plt.ion()
        fig,axes = plt.subplots(figsize=(15, 8),nrows=3,ncols=2)
        for steps in range(1,max_iters+1):

            # 获取trans_alpha
            trans_alpha = steps/max_iters

            # 输入标准正态分布
            z = np.random.normal(size=(batch_size, latents_size))
            # 获取数据集
            minibatch = sess.run(data_batch)
            # 格式修正
            minibatch = np.reshape(minibatch,[-1,3,sl]).astype(np.float32)
            if isTransit:
                # 对当前数据集先缩小一倍再放大一倍，提取上一级样本信息
                minibatch_low = utilities.downsampling_and_upsampling_batch(minibatch)
                minibatch_input = trans_alpha * minibatch + (1 - trans_alpha) * minibatch_low  # 数据集过渡处理
            else:
                minibatch_input = minibatch
            # 修正为tf格式
            minibatch_input = np.reshape(minibatch_input,[-1,3,sl,1])

            # 训练判别器
            for i in range(n_critic):
                sess.run(d_train_opt, feed_dict={real_SS: minibatch_input, latents: z})
            # 训练生成器
            sess.run(g_train_opt, feed_dict={latents: z})

            # recording training_losses
            train_loss_d = sess.run(d_loss, feed_dict={real_SS: minibatch_input, latents: z})
            train_loss_g = sess.run(g_loss, feed_dict={latents: z})
            Wasserstein = sess.run(Wass, feed_dict={real_SS: minibatch_input, latents: z})
            SLOPES = sess.run(slopes_m, feed_dict={real_SS: minibatch_input, latents: z})

            # recording training_products
            z = np.random.normal(size=(10, latents_size))
            gen_SS = sess.run(fake_SS, feed_dict={latents: z}).reshape([-1,3,sl])

            # 实时PLOT
            if steps % 500 == 0:
            #     batch_fft = utilities.batch_fft_mean(gen_SS)
                runtime_showing(steps,isTransit,sl,axes,minibatch[0][0],minibatch[0][1],minibatch[0][2],
                                gen_SS[0][0],gen_SS[0][1],gen_SS[0][2])

            # 打印信息
            print_traing_info(steps, max_iters, level, sl, isTransit, train_loss_d, train_loss_g, Wasserstein, SLOPES)

            #  记录训练信息
            if steps % 10 == 0:
                # （1）记录损失函数
                losses.append([steps, train_loss_d, train_loss_g])
                # （2）记录生成样本
                GenLog.append(gen_SS)

                # （3）保存生成模型
            if steps % 1000 == 0:
                saver.save(sess, model_path + '/network.ckpt', global_step=steps)  # 保存模型

            # 计算swd
            # if steps % 1000 == 0 and level>1:
            #     FAKES = [] # 2000
            #     for i in range(40):
            #         z = np.random.normal(size=[50, latents_size])
            #         fakes = sess.run(fake_eegs, feed_dict={latents: z})
            #         fakes = np.reshape(fakes,[-1,26,sl])
            #         FAKES.append(fakes)
            #     FAKES = np.concatenate(FAKES,axis=0)
            #     BD = SWDs.minibatch_band(FAKES)
            #     d_desc = SWDs.get_descriptors_for_minibatch(BD, 7,64)
            #     d_desc = SWDs.finalize_descriptors(d_desc)
            #     swd = SWDs.avg_sliced_wasserstein(d_desc, DESC[str(sl)], 4, 64) * 1e3
            #     SWD.append(swd)
            #     print('当前生成样本swd(x1e3):', swd,'...')

        # 关闭线程
        coord.request_stop()
        coord.join(threads)
        # plt.close(fig)

        # 计时结束：
        time_end = time.time()
        print('迭代结束，耗时：%.2f秒' % (time_end - time_start))

        # 保存信息
        Saving_Train_Log('losses_%d_trans_%s' % (level, isTransit), losses)
        Saving_Train_Log('WASS_%d_trans_%s' % (level, isTransit), WASS)
        Saving_Train_Log('GenLog_%d_trans_%s' % (level, isTransit), GenLog)

    # 清理图
    tf.reset_default_graph()

if __name__ == '__main__':

    latents_size = 200\


    batch_size = 32
    data_size = 10000
    epochs = 20
    lowest = 0
    highest = 5

    # progressive growing
    # PGGAN(latents_size, batch_size, lowest, highest, level=0, isTransit=False,data_size=data_size,epochs=epochs)
    # PGGAN(latents_size, batch_size, lowest, highest, level=1, isTransit=True, data_size=data_size, epochs=epochs)
    # PGGAN(latents_size, batch_size, lowest, highest, level=1, isTransit=False, data_size=data_size, epochs=epochs)
    # PGGAN(latents_size, batch_size, lowest, highest, level=2, isTransit=True, data_size=data_size, epochs=epochs)
    # PGGAN(latents_size, batch_size, lowest, highest, level=2, isTransit=False, data_size=data_size, epochs=epochs)
    PGGAN(latents_size, batch_size, lowest, highest, level=3, isTransit=True, data_size=data_size, epochs=epochs)
    PGGAN(latents_size, batch_size, lowest, highest, level=3, isTransit=False, data_size=data_size, epochs=epochs)
    PGGAN(latents_size, batch_size, lowest, highest, level=4, isTransit=True, data_size=data_size, epochs=epochs)
    PGGAN(latents_size, batch_size, lowest, highest, level=4, isTransit=False, data_size=data_size, epochs=epochs)
    PGGAN(latents_size, batch_size, lowest, highest, level=5, isTransit=True, data_size=data_size, epochs=epochs)
    PGGAN(latents_size, batch_size, lowest, highest, level=5, isTransit=False, data_size=data_size, epochs=epochs)
