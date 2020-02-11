# coding:utf-8
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import datetime
#------------------------
from utils import *
import cfgs_hw as cfgs
#------------------------

# 打印模型相关信息
def display_cfgs(models):
    print('global_cfgs')
    cfgs.showcfgs(cfgs.global_cfgs)
    print('dataset_cfgs')
    cfgs.showcfgs(cfgs.dataset_cfgs)
    print('net_cfgs')
    cfgs.showcfgs(cfgs.net_cfgs)
    print('optimizer_cfgs')
    cfgs.showcfgs(cfgs.optimizer_cfgs)
    print('saving_cfgs')
    cfgs.showcfgs(cfgs.saving_cfgs)
    for model in models:
        print(model)

# 展开label
def flatten_label(target):
    label_flatten = []
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0)+1]
        label_length.append(cur_label.index(0)+1)
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    return (label_flatten, label_length)

def Train_or_Eval(models, state = 'Train'):
    for model in models:
        if state == 'Train':
            # 启用 BatchNormalization 和 Dropout
            model.train()
        else:
            # 不启用 BatchNormalization 和 Dropout
            model.eval()

# 导数清零
def Zero_Grad(models):
    for model in models:
        model.zero_grad()

def Updata_Parameters(optimizers, frozen):
    for i in range(0, len(optimizers)):
        if i not in frozen:
            # 每次更新完权重之后，导数清零
            optimizers[i].step()

#---------------------dataset
# load data
def load_dataset():
    train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
    train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train']) 

    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])
    # pdb.set_trace()
    return (train_loader, test_loader)

#---------------------network
# 模型初始化
def load_network():
    model_fe = cfgs.net_cfgs['FE'](**cfgs.net_cfgs['FE_args'])
    # FE层输出的size，传给CAM层的参数
    cfgs.net_cfgs['CAM_args']['scales'] = model_fe.Iwantshapes()
    model_cam = cfgs.net_cfgs['CAM'](**cfgs.net_cfgs['CAM_args'])

    model_dtd = cfgs.net_cfgs['DTD'](**cfgs.net_cfgs['DTD_args'])

    # 是否加载预训练的特征提取器、卷积对齐模块、去耦解码器网络参数
    if cfgs.net_cfgs['init_state_dict_fe'] != None:
        model_fe.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_fe']))
    if cfgs.net_cfgs['init_state_dict_cam'] != None:
        model_cam.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_cam']))
    if cfgs.net_cfgs['init_state_dict_dtd'] != None:
        model_dtd.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_dtd']))
    
    model_fe.cuda()
    model_cam.cuda()
    model_dtd.cuda()
    return (model_fe, model_cam, model_dtd)
#----------------------optimizer
# 优化器
def generate_optimizer(models):
    out = []
    scheduler = []
    # PE CAM DTD三层采用的优化器，可以各采用不同优化器
    # pytorch中optimizer.step()用在每个mini-batch之中，而scheduler.step()用在epoch里面
    for i in range(0, len(models)):
        out.append(cfgs.optimizer_cfgs['optimizer_{}'.format(i)](
                    models[i].parameters(),
                    **cfgs.optimizer_cfgs['optimizer_{}_args'.format(i)]))
        scheduler.append(cfgs.optimizer_cfgs['optimizer_{}_scheduler'.format(i)](
                    out[i],
                    **cfgs.optimizer_cfgs['optimizer_{}_scheduler_args'.format(i)]))
    return tuple(out), tuple(scheduler)
#---------------------testing stage
def test(test_loader, model, tools):
    Train_or_Eval(model, 'Eval')
    for sample_batched in test_loader:
        data = sample_batched['image']
        label = sample_batched['label']
        target = tools[0].encode(label)

        data = data.cuda()
        target = target
        label_flatten, length = tools[1](target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # 特征提取器（FE）
        features= model[0](data)
        #卷积对齐模块（CAM）
        A = model[1](features)
        # 去耦解码器（DTD）
        output, out_length = model[2](features[-1], A, target, length, True)
        # 测试准确率
        tools[2].add_iter(output, out_length, length, label)
    tools[2].show()
    # ??暂时不懂为什么最后还运行训练
    Train_or_Eval(model, 'Train')
#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    # prepare nets, optimizers and data
    model = load_network()
    # 显示模型相关信息
    display_cfgs(model)
    optimizers, optimizer_schedulers = generate_optimizer(model)
    criterion_CE = nn.CrossEntropyLoss().cuda()
    train_loader, test_loader = load_dataset()
    print('preparing done')
    # --------------------------------
    # prepare tools
    train_acc_counter = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    # 每隔show_interval显示
    loss_counter = Loss_counter(cfgs.global_cfgs['show_interval'])
    # 编码解码类初始化，case_sensitive大小写敏感
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    #---------------------------------
    if cfgs.global_cfgs['state'] == 'Test':
        test((test_loader), 
             model, 
            [encdec,
             flatten_label,
             test_acc_counter])
        exit()
    # --------------------------------
    total_iters = len(train_loader)
    for nEpoch in range(0, cfgs.global_cfgs['epoch']):
        for batch_idx, sample_batched in enumerate(train_loader):
            # data prepare
            data = sample_batched['image']
            label = sample_batched['label']
            target = encdec.encode(label)
            Train_or_Eval(model, 'Train')
            data = data.cuda()
            label_flatten, length = flatten_label(target)
            target, label_flatten = target.cuda(), label_flatten.cuda()
            # net forward
            # FE、CAM、DTD的前向传播
            features = model[0](data)           
            A = model[1](features)
            output, attention_maps = model[2](features[-1], A, target, length)
            # computing accuracy and loss
            # 计算正确数量及字符距离，词距离
            train_acc_counter.add_iter(output, length.long(), length, label)
            # 损失
            loss = criterion_CE(output, label_flatten)
            loss_counter.add_iter(loss)
            # update network
            # 每次更新完权重之后，导数清零
            Zero_Grad(model)
            loss.backward()
            # 梯度裁剪，防止梯度过大
            nn.utils.clip_grad_norm_(model[0].parameters(), 20, 2)
            nn.utils.clip_grad_norm_(model[1].parameters(), 20, 2)
            nn.utils.clip_grad_norm_(model[2].parameters(), 20, 2)
            # optimizer用于minibatch
            Updata_Parameters(optimizers, frozen = [])
            # visualization and saving
            # 每隔show_interval打印训练准确率，损失相关信息
            if batch_idx % cfgs.global_cfgs['show_interval'] == 0 and batch_idx != 0:
                print(datetime.datetime.now().strftime('%H:%M:%S'))
                print('Epoch: {}, Iter: {}/{}, Loss dan: {}'.format(
                                    nEpoch,
                                    batch_idx,
                                    total_iters,
                                    loss_counter.get_loss()))
                train_acc_counter.show()
            # 每隔test_interval跑验证机，打印准确率相关信息
            if batch_idx % cfgs.global_cfgs['test_interval'] == 0 and batch_idx != 0:
                test((test_loader), 
                     model, 
                    [encdec,
                     flatten_label,
                     test_acc_counter])

            # 训练完保存模型
            if nEpoch % cfgs.saving_cfgs['saving_epoch_interval'] == 0 and \
               batch_idx % cfgs.saving_cfgs['saving_iter_interval'] == 0 and \
               batch_idx != 0:
                for i in range(0, len(model)):
                    torch.save(model[i].state_dict(),
                             cfgs.saving_cfgs['saving_path'] + 'E{}_I{}-{}_M{}.pth'.format(
                                nEpoch, batch_idx, total_iters, i))
        # optimizer_schedulers用于epoch
        Updata_Parameters(optimizer_schedulers, frozen = [])
