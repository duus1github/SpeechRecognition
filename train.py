#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:train.py
@time:2022/05/24
@desc:''

"""
import argparse
import visdom
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from resnet import ResNet18
from speech_data import SpeechData

parser = argparse.ArgumentParser(description="Run SpeechRecognition.")
parser.add_argument('--batch_size', nargs='?', default=64,
                    help='number of samples in one batch')
parser.add_argument('--epochs', nargs='?', default=10,  # 150
                    help='number of epochs in SGD')
parser.add_argument('--lr', nargs='?', default=1e-3,
                    help='learning rate for the SGD')
parser.add_argument('--device', nargs='?', default='cpu',
                    help='training device')
args = parser.parse_args()

# 设置种子点
torch.manual_seed(1234)
# todo:导入数据
train_db = SpeechData('dataSet', 224, mode='train')
val_db = SpeechData('dataSet', 224, mode='val')
test_db = SpeechData('dataSet', 224, mode='test')
# todo:适用dataloader,处理数据
train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=args.batch_size)
test_loader = DataLoader(test_db, batch_size=args.batch_size)


def evalute(model, loader):
    """
    计算预测正确数据和错误数据比例
    :param model:
    :param loader:
    :return:
    """
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def main():
    viz = visdom.Visdom()
    model = ResNet18(5).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_loss'))
    global_step = 0
    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)

            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
        # todo:记录并保存最好的模型
        if epoch % 1 == 0:
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best.model')
                viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc:', best_acc, 'best_epoch:', best_epoch)
    # todo:加载最好的模型
    model.load_state_dict(torch.load('best.model'))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc.', test_acc)


if __name__ == '__main__':
    main()
