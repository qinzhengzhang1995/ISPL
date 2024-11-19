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
kuaipais = 2
fenpi = 2
meipi = int(kuaipais/fenpi)
print('kuaipais',kuaipais)
print('fenpi',fenpi)
print('meipi',meipi)
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
#############################################################################33





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
trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batchsize, shuffle=True, drop_last=True)
testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batchsize, shuffle=True, drop_last = True)

# trainsmallloader = DataLoader(TensorDataset(X_train_small, y_train_small), batch_size=batchsize, shuffle=True)
# testsmallloader = DataLoader(TensorDataset(X_test_small, y_test_small), batch_size=batchsize, shuffle=True)




# 加载数据集
# transform = transforms.Compose([transforms.ToTensor()])
#
# train_loader_teacher = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
# train_loader_student = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=32, shuffle=True)

# 初始化教师模型和学生模型
# teacher_model = DML(7)
# student_model = DML(7)

# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer_teacher = optim.SGD(teacher_model.parameters(), lr=0.01)
# optimizer_student = optim.SGD(student_model.parameters(), lr=0.01)
#
# # 训练模型
# epochs = 10
# for epoch in range(epochs):
#     for (data_teacher, target_teacher), (data_student, target_student) in zip(train_loader_teacher,
#                                                                               train_loader_student):
#         # 教师模型训练
#         optimizer_teacher.zero_grad()
#         output_teacher = teacher_model(data_teacher)
#         loss_teacher = criterion(output_teacher, target_teacher)
#         loss_teacher.backward()
#         optimizer_teacher.step()
#
#         # 学生模型训练
#         optimizer_student.zero_grad()
#         output_student = student_model(data_student)
#         loss_student = criterion(output_student, target_student)
#         loss_student.backward()
#         optimizer_student.step()
#
#         # 更新学生模型参数
#         for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
#             student_param.data = teacher_param.data
#
#     print(f'Epoch {epoch + 1}, Teacher Loss: {loss_teacher.item()}, Student Loss: {loss_student.item()}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################教师model###################################################################
model = DML(7)
model = CustomAlexNet(7)
# if torch.cuda.device_count() > 1:
# model = nn.DataParallel(model,device_ids=[0,1])
model.to(device)
###########################学生model#################################################################
model_student = DML(7)
# model_student = CustomAlexNet(7)
# if torch.cuda.device_count() > 1:
# model_student = nn.DataParallel(model_student,device_ids=[0,1])
model_student.to(device)
##############teacher优化############################
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()
###############学生优化################
optimizer_student = torch.optim.Adam(model_student.parameters(), lr=1e-4, weight_decay=1e-6)
temp = 5  # 蒸馏温度
hard_loss = nn.CrossEntropyLoss()
alpha = 0.5  # hard_loss权重
# soft_loss = nn.KLDivLoss(reduction='batchmean')
# soft_loss = nn.KLDivLoss()

best_acc = 0.0
best_acc_student = 0.0
epochs = 100
for epoch in range(epochs):
    train_loss = 0.0
    train_loss_student = 0.0
    train_acc =0.0
    train_acc_student = 0.0
    test_loss = 0.0
    test_loss_student =0.0
    test_acc = 0.0
    test_acc_student = 0.0
    model.train()  # 教师训练模式
    model_student.train()  ##学生训练模式
    batch_jishu_xunlian = 0
    for data_x, data_y in trainloader:
        # print('data_x', data_x.shape, data_x_small.shape)
        # print('data_y',data_y,data_y_small)
        batch_jishu_xunlian = batch_jishu_xunlian + 1
        preds = model(data_x[:,:,:,0:kuaipais].float().to(device)) # 前向传播得到预测结果    教师的预测
        total_student_preds = 0
        for kuaipai in range(int(kuaipais/meipi)):
            # print('kuaipai',kuaipai)
            # print('datax',data_x.shape)
            students_preds = model_student(data_x[:,:,:,meipi*kuaipai:meipi*(kuaipai+1)].to(device).float())  #####学生的预测
            train_acc_student += np.sum(np.argmax(students_preds.cpu().detach().numpy(), axis=1) == data_y.numpy())
            total_student_preds = total_student_preds + students_preds
        arg_student_preds = total_student_preds/(kuaipais/meipi)


        teacher_hard_loss = criterion(preds, data_y.long().to(device)) #########教师loss
        teacher_KL = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(preds / temp, dim=1),F.softmax(arg_student_preds.detach() / temp, dim=1)) * (temp * temp)
        loss = alpha * teacher_hard_loss + (1-alpha) * teacher_KL

        students_loss = hard_loss(arg_student_preds, data_y.long().to(device))
        D_KL = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(arg_student_preds / temp, dim=1), F.softmax(preds.detach() / temp, dim=1)) * (temp * temp)
        loss_student_total_batch = alpha * students_loss + (1 - alpha) * D_KL   ###学生总loss

        #####
        train_acc += np.sum(np.argmax(preds.cpu().detach().numpy(), axis=1) == data_y.numpy())



        train_loss += loss.item()
        ######
        optimizer.zero_grad()  # 清空梯度信息
        loss.backward()  # 损失反向传播
        optimizer.step()  # 对网络参数进行优化
        ####
        # train_acc_student += np.sum(np.argmax(students_preds.cpu().detach().numpy(), axis=1) == data_y_small.numpy())
        train_loss_student += loss_student_total_batch.item()  ####我们这里的loss同比记录和的loss
        #######
        optimizer_student.zero_grad()  # 清空梯度信息
        loss_student_total_batch.backward()  # 损失反向传播  反向传播的loss 是两个loss的和
        optimizer_student.step()  # 对网络参数进行优化

    loss_final_train = train_loss / batch_jishu_xunlian / batchsize
    acc_final_train  =train_acc / batch_jishu_xunlian / batchsize

    loss_final_train_student = train_loss_student / batch_jishu_xunlian / batchsize
    acc_final_train_student = train_acc_student / batch_jishu_xunlian / batchsize / (kuaipais/meipi)
    # loss_final_train = train_loss / len(trainloader)/batch_size
    # acc_final_train = train_acc / len(trainloader)/batch_size
    print('epoch: %3d\n train acc: %3.6f, train loss: %3.6f' %(epoch+1, acc_final_train,loss_final_train))
    print('student train acc: %3.6f,student train loss: %3.6f' %(acc_final_train_student,loss_final_train_student))
        # 进入测试模式
    model.eval()
    model_student.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_loss_student = 0.0
    test_acc_student = 0.0
    with torch.no_grad():  # 固定所有参数的梯度为0，因为测试阶段不需要进行优化
        batch_jishu_ceshi = 0
        for data_x_test, data_y_test in testloader:
            batch_jishu_ceshi = batch_jishu_ceshi + 1


            teachers_preds_test = model(data_x_test[:,:,:,0:kuaipais].to(device).float())   ###教师的预测
            total_student_preds_test = 0
            for kuaipai_test in range(int(kuaipais/meipi)):
                # print('kuaipai_test', kuaipai_test)
                students_preds_test = model_student(data_x_test[:, :, :, meipi*kuaipai_test:meipi*(kuaipai_test+1)].to(device).float())  #####学生的预测
                test_acc_student += np.sum(np.argmax(students_preds_test.cpu().detach().numpy(), axis=1) == data_y_test.numpy())
                total_student_preds_test = total_student_preds_test + students_preds_test
            arg_student_preds_test = total_student_preds_test / (kuaipais/meipi)  #####学生的预测
            # students_preds_test = model_student(data_x_test_small.float().to(device))
            ###教师的loss与计算
            loss_t_batch = criterion(teachers_preds_test, data_y_test.long().to(device))
            teacher_KL_test = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(teachers_preds_test / temp, dim=1), F.softmax(arg_student_preds_test.detach() / temp, dim=1)) * (temp * temp)
            loss_test = alpha * loss_t_batch + (1 - alpha) * teacher_KL_test

            test_acc += np.sum(np.argmax(teachers_preds_test.cpu().detach().numpy(), axis=1) == data_y_test.numpy())
            test_loss += loss_test.item()
            ###学生的loss与计算
            students_loss_test = hard_loss(arg_student_preds_test, data_y_test.long().to(device))
            # ditillation_loss_test = soft_loss(
            #     F.softmax(students_preds_test / temp, dim=1),
            #     F.softmax(teachers_preds_test.detach() / temp, dim=1)
            # )
            D_KL_test = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(arg_student_preds_test / temp, dim=1),F.softmax(teachers_preds_test.detach() / temp, dim=1)) * (temp * temp)
            loss_he = alpha * students_loss_test + (1 - alpha) * D_KL_test
            #####是否用平均值来求准确率
            # test_acc_student += np.sum(np.argmax(arg_student_preds_test.cpu().detach().numpy(), axis=1) == data_y_test.numpy())
            test_loss_student += loss_he.item()
        loss_final_test = test_loss / batch_jishu_ceshi / batchsize
        acc_final_test = test_acc / batch_jishu_ceshi / batchsize
        loss_final_test_student = test_loss_student / batch_jishu_ceshi / batchsize
        acc_final_test_student = test_acc_student / batch_jishu_ceshi / batchsize / (kuaipais/meipi)
        # loss_final_test = test_loss / len(testloader)/batch_size
        # acc_final_test = test_acc / len(testloader)/batch_size
        print('test acc: %3.6f, test loss:%3.6f' % (acc_final_test,loss_final_test))
        print('student test acc: %3.6f, student test loss:%3.6f' % (acc_final_test_student, loss_final_test_student))
