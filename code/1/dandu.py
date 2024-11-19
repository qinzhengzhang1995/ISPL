import os

# from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# from matrixDataset import matrixDataset_underwater
from mymodel import Classifier_singal_out, LeNet5, chenbodata
# from pinjie import data_pinjie
# from read_file import readfile_output, readfile_input
import random


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from mymodel import DML,CustomAlexNet
import random
import os
import numpy as np
from read_file import readfile_output, readfile_input
from pinjie import data_pinjie
from torch.utils.data import DataLoader, TensorDataset


def seed_it(seed):
    random.seed(seed) #可以注释掉
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #这个懂吧
    torch.backends.cudnn.deterministic = True #确定性固定
    torch.backends.cudnn.benchmark = False #False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  #增加运行效率，默认就是True
    torch.manual_seed(seed)
seed_it(1)

# CUDA_VISIBLE_DEVICES=0,1

# train = datasets.MNIST('data/', download=True, train=True)
# test = datasets.MNIST('data/', download=True, train=False)
batchsize =  64
kuaipais = 10
print('kuaipais',kuaipais)
print('kuaipais/int/',int(kuaipais/1))
################################################################################################################
# X_train = train.data.unsqueeze(1)/255.0
# y_train = train.targets
# print('X_train',X_train.shape)
# print('y_train',y_train.shape)
#
# trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batchsize, shuffle=True, drop_last= True)
#
# X_test = test.data.unsqueeze(1)/255.0
# y_test = test.targets
#
# testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batchsize, shuffle=True, drop_last= True)
#############################################################################################################


#####train ######
list_title_train = os.listdir('.//train')
url_train = './/train//'
type1 = 'x_train'
X_train= readfile_input(list_title_train,url_train,type1)
X_train=X_train.transpose(3,2,1,0)
print('total_final_input',X_train.shape)
##裁剪X
X_train = X_train[:,:,0:100,:]
print('total_final_input_2',X_train.shape)
#####大小存一下

####拼成CNN输入
X_train = data_pinjie(X_train)
print('X_train',X_train.shape)
X_train = X_train.transpose(2,0,1)
new_train_shape = (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_train = X_train.reshape(new_train_shape)
X_train = torch.tensor(X_train)
print('X_train_after_reshape',X_train.shape)


#############
type2 = 'y1_train'
y_train = readfile_output(list_title_train, url_train, type2)
print('total_final_output',y_train.shape)
#######裁剪y#####
y1 = y_train[0:100,:]
y2 = y_train[1000:1100,:]
y3 = y_train[2000:2100,:]
y4 = y_train[3000:3100,:]
y5 = y_train[4000:4100,:]
y6 = y_train[5000:5100,:]
y7 = y_train[6000:6100,:]
y_train=np.concatenate([y1,y2,y3,y4,y5,y6,y7])
print('total_final_output_2',y_train.shape)
# print(y_train)
#########################
y_train = y_train.squeeze(-1)
y_train = torch.tensor(y_train)



#############

##########################################################################################







###############test#################
list_title_test = os.listdir('.//test')
url_test = './/test//'

type3 = 'x_test'
X_test= readfile_input(list_title_test,url_test,type3)
X_test=X_test.transpose(3,2,1,0)
print('total_final_input',X_test.shape)

####拼成CNN输入
X_test = data_pinjie(X_test)
X_test = X_test.transpose(2,0,1)

new_test_shape = (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
X_test = X_test.reshape(new_test_shape)
X_test = torch.tensor(X_test)
print('X_test_after_reshape',X_test.shape)
#############

type4 = 'y1_test'
y_test = readfile_output(list_title_test, url_test, type4)
print('total_final_output',y_test.shape)
y_test = y_test.squeeze(-1)
y_test = torch.tensor(y_test)
######################



# 定义教师模型和学生模型的网络结构
trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batchsize, shuffle=True)
testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batchsize, shuffle=True)

# trainsmallloader = DataLoader(TensorDataset(X_train_small, y_train_small), batch_size=batchsize, shuffle=True)
# testsmallloader = DataLoader(TensorDataset(X_test_small, y_test_small), batch_size=batchsize, shuffle=True)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




#####################


##########################3


################不需要教师model了####################
# Mymodel = torch.load('.//before_best_5kuaipai_new.pth')
model_new = DML(7)

# model_new = CustomAlexNet(7)   ##改网络需要同时改学习率
# newout= 10
print(model_new)
model_new.to(device)
opt = torch.optim.Adam(model_new.parameters(), lr=1e-4, weight_decay=1e-6)    ###  DML为1E-4   ，   CustomAlexNet 为2e-5
# 这部分仅仅是为了展示单独训练一个学生模型时的效果，与采用蒸馏训练对比一下
criterion = nn.CrossEntropyLoss()

epochs = 100
best_acc_student = 0.0
for epoch in range(epochs):
    train_loss = 0.0
    train_acc =0.0
    test_loss = 0.0
    test_acc = 0.0
    model_new.train()  # 训练模式
    batch_jishu_xunlian = 0
    for data_x, data_y in trainloader:

        batch_jishu_xunlian = batch_jishu_xunlian + 1
        preds = model_new(data_x[:, :, :, 0:kuaipais].float().to(device)) # 前向传播得到预测结果
        loss = criterion(preds, data_y.long().to(device))
        train_acc += np.sum(np.argmax(preds.cpu().detach().numpy(), axis=1) == data_y.numpy())
        train_loss += loss.item()
        opt.zero_grad()  # 清空梯度信息
        loss.backward()  # 损失反向传播
        opt.step()  # 对网络参数进行优化
    loss_final_train = train_loss / batch_jishu_xunlian / batchsize
    acc_final_train  =train_acc / batch_jishu_xunlian / batchsize
    # loss_final_train = train_loss / len(trainloader)/batch_size
    # acc_final_train = train_acc / len(trainloader)/batch_size
    print('epoch: %3d\n train acc: %3.6f, train loss: %3.6f' %(epoch+1, acc_final_train,loss_final_train))
        # 进入测试模式
    model_new.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():  # 固定所有参数的梯度为0，因为测试阶段不需要进行优化
        batch_jishu_ceshi = 0
        for data_x_test, data_y_test in testloader:
            batch_jishu_ceshi = batch_jishu_ceshi + 1
            pred_test = model_new(data_x_test[:, :, :, 0:kuaipais].float().to(device))  # 前向传播得到测试结果，preds为一个向量
            loss_t_batch = criterion(pred_test, data_y_test.long().to(device))
            test_acc += np.sum(np.argmax(pred_test.cpu().detach().numpy(), axis=1) == data_y_test.numpy())
            test_loss += loss_t_batch.item()
        loss_final_test = test_loss / batch_jishu_ceshi / batchsize
        acc_final_test = test_acc / batch_jishu_ceshi / batchsize
        # loss_final_test = test_loss / len(testloader)/batch_size
        # acc_final_test = test_acc / len(testloader)/batch_size
        print('test acc: %3.6f, test loss:%3.6f' % (acc_final_test,loss_final_test))
    if acc_final_test > best_acc_student:
        best_acc_student = acc_final_test
        # model_student_name = 'before_best_5kuaipai_new.pth'
        # torch.save(model_new, model_student_name)
        # student_model_scratch = model_new   #把学生模型存下来
        print('best_epoch_student',epoch)
    else:
        best_acc_student=best_acc_student
    # if epoch % 1 == 0:
    #     model_student_name = os.path.join('student_%depoch.pth' % (epoch))
    #     torch.save(model, model_student_name)





