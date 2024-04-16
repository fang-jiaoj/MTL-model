import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import  BertModel
from dataset import Smiles_Bert_Dataset,Pretrain_Collater
import time
import os
from torch.utils.data import DataLoader
from metrics import AverageMeter
import argparse

#使用 Python 的 argparse 库创建了一个命令行参数解析器 parser，并定义了一个命令行参数 --Smiles_head
parser = argparse.ArgumentParser() #创建一个参数解析器
parser.add_argument('--Smiles_head', nargs='+', default=["CAN_SMILES"], type=str)
#使用.add_argument()方法在参数解析器 parser 中添加一个参数的操作
args = parser.parse_args() #解析命令行参数的语句，它会解析命令行中传递的参数，并将它们存储在变量 args 中

#指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#三个不同的模型架构，分别为 small、medium 和 large，每个架构具有不同的参数设置，例如层数、头数和模型维度，以及模型权重的保存路径
small = {'name': 'small', 'num_layers': 4, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights'}
medium = {'name': 'medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights'}
large = {'name': 'large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights'}

arch = medium     ## small 3 4 128   medium: 6 6  256     large:  12 8 516
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']

dff = d_model*4 #隐藏层维度
vocab_size = 60 #词汇表大小
dropout_rate = 0.1 #drop_out率

#它创建了一个 BertModel 模型，使用了上面提取的架构参数，并将模型移动到指定的计算设备（device）上
model = BertModel(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,vocab_size=vocab_size)
model.to(device)

# data = pd.read_csv('data/chem.csv')

#返回经过mask的序列，原始序列以及哪些位置进行了MASK操作
full_dataset = Smiles_Bert_Dataset('chem.csv',Smiles_head=args.Smiles_head)

#它将 full_dataset 分为训练集和测试集，其中训练集占总数据的 90%，测试集占 10%
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

#它创建了训练集和测试集的数据加载器（train_dataloader 和 test_dataloader），用于批量加载数据进行训练和测试
train_dataloader = DataLoader(train_dataset,batch_size=512,shuffle=True,collate_fn=Pretrain_Collater())
test_dataloader = DataLoader(test_dataset,batch_size=512,shuffle=False,collate_fn=Pretrain_Collater())

#Adam 优化器，用于更新模型的参数
optimizer = optim.Adam(model.parameters(),1e-4,betas=(0.9,0.98))

#它定义了损失函数 loss_func，这里使用了交叉熵损失函数，其中 ignore_index=0 表示忽略标签为 0 的部分，reduction='none' 表示不对损失进行汇总
loss_func = nn.CrossEntropyLoss(ignore_index=0,reduction='none')

#定义了用于计算训练和测试过程中损失和准确率的平均值的 AverageMeter 类
# train_loss = AverageMeter()
# train_acc = AverageMeter()
# test_loss = AverageMeter()
# test_acc = AverageMeter()

#修改成以下：
# batch记录
train_loss = AverageMeter()
train_acc = AverageMeter()
test_loss = AverageMeter()
test_acc = AverageMeter()

# epoch记录
epoch_train_loss = AverageMeter()
epoch_train_acc = AverageMeter()
epoch_test_loss = AverageMeter()
epoch_test_acc = AverageMeter()

def train_step(x, y, weights):
    """该函数定义了训练步骤"""
    model.train() #将模型设为训练模式
    optimizer.zero_grad() #将梯度清零，防止梯度累积
    predictions = model(x) #得到模型的预测结果
    loss = (loss_func(predictions.transpose(1,2),y)*weights).sum()/weights.sum() #仅计算被MASK位置的损失值
    #计算了模型的预测结果 predictions 与实际标签 y 之间的损失，prediction.transpose(1,2)的维度（batch_size,vocab_size,src_len），y的维度（batch_size,src_len）
    #weights的维度（batch_size,src_len）,其中被MASK的位置值为1，否则为0，整个表达式的结果是计算了加权损失的平均值
    loss.backward() #进行反向传播
    optimizer.step() #更新模型的参数

    train_loss.update(loss.detach().item(),x.shape[0]) #更新训练过程中的损失值,平均损失值
    #loss.detach().item()是当前batch的损失值,x.shape[0]表示当前batch的样本数量
    train_acc.update(((y==predictions.argmax(-1))*weights).detach().cpu().sum().item()/weights.cpu().sum().item(),
                     weights.cpu().sum().item()) #更新训练过程中的准确率，平均准确率，只关注被MASK的位置的准确率
    ##predictions.argmax(-1)就是在最后一个维度上找到一行中最大值的索引（这个索引就是被MASK的值），返回维度（batch_size，src_len）
    #y == predictions.argmax(-1) 的操作会逐元素比较模型的预测结果和真实标签是否相等。它返回一个布尔张量
    #只关注被MASK位置的值是否匹配，.detach()用于分离张量，.item() 将结果转换为标量
    # ((y==predictions.argmax(-1))*weights).detach().cpu().sum().item()得到被MASK位置预测正确的数量



def test_step(x,y, weights):
    """定义一个测试步骤"""
    model.eval() #将模型设为评估模式
    with torch.no_grad(): #禁用梯度计算，加快运行速度，节省内存
        predictions = model(x)
        loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum()/weights.sum() #计算被MASK位置的损失值

        test_loss.update(loss.detach(), x.shape[0]) #更新损失值和平均损失值
        test_acc.update(((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item()/weights.cpu().sum().item(),
                              weights.cpu().sum().item()) #更新MASK位置的预测准确率和平均准确率

###进行一个训练循环，它会在数据集上训练模型，并定期评估模型的性能。
for epoch in range(100): #进行100次训练
    start = time.time()  #记录当前训练周期的开始时间

    #将训练数据和测试数据都按照batch进行训练和测试
    for (batch, (x, y, weights)) in enumerate(train_dataloader):
        train_step(x, y, weights)

        if batch%500==0: #是否达到了每 500 个批次打印一次的条件，如果是，则打印训练损失和准确性
            print('Epoch {} Batch {} training Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.avg))
            print('traning Accuracy: {:.4f}'.format(train_acc.avg))
            # epoch记录
            epoch_train_loss.update(train_loss.avg)
            epoch_train_acc.update(train_acc.avg)
            # batch清零 train
            train_acc.reset()
            train_loss.reset()


        if batch % 1000 == 0: #这一行检查是否达到了每 1000 个批次执行一次测试的条件
            for x, y ,weights in test_dataloader:
                test_step(x, y , weights) #打印测试损失和准确性：在测试数据集上计算损失和准确性，并打印它们
            print('Test loss: {:.4f}'.format(test_loss.avg))
            print('Test Accuracy: {:.4f}'.format(test_acc.avg))
            # epoch记录
            epoch_test_acc.update(test_acc.avg)
            epoch_test_loss.update(test_loss.avg)
            # batch清零 test
            test_acc.reset()
            test_loss.reset()

            # #将前一个batch的acc值和loss值更新为初始值，从而不断累积，计算一个epoch的平均损失和平均acc值
            # test_acc.reset()
            # test_loss.reset()
            # train_acc.reset()
            # train_loss.reset()

    print('Epoch {} is Done!'.format(epoch))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    print('Epoch {} Training Loss {:.4f}'.format(epoch + 1, epoch_train_loss.avg)) #一次训练的平均损失
    print('training Accuracy: {:.4f}'.format(epoch_train_acc.avg)) #一次训练的平均acc-->SMILES恢复率
    print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, epoch_test_loss.avg)) #测试集的平均损失
    print('test Accuracy: {:.4f}'.format(epoch_test_acc.avg)) #一次测试的平均acc
    torch.save(model.state_dict(),'weights/' + arch['path']+'_bert_weights{}_{}.pt'.format(arch['name'],epoch+1) )
    #将整个模型一个epoch的状态参数（权重/偏置）存储到.pt文件中
    torch.save(model.encoder.state_dict(), 'weights/' + arch['path'] + '_bert_encoder_weights{}_{}.pt'.format(arch['name'], epoch + 1))
    #将模型中的encoder层的状态参数（权重/偏置）存储到.pt文件中
    print('Successfully saving checkpoint!!!')

    ###修改
    epoch_train_acc.reset()
    epoch_test_acc.reset()
    epoch_train_loss.reset()
    epoch_test_loss.reset()



