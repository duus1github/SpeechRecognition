#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:speech_data.py
@date:2023/3/2 20:44
@desc:''
"""
import torch
import os, glob
import random, csv

from PIL import Image as imim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SpeechData(Dataset):
    def __init__(self, root, resize, mode):
        super(SpeechData, self).__init__()

        self.root = root  # 路径
        self.resize = resize

        # 将名称对应为label
        self.name2label = {}

        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            else:
                self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        self.images, self.labels = self.load_csv('images.csv')
        # todo:取数据
        if mode == 'train':  # 60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':  # 20% 60%->80%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:  # 测试数据，也是取20%：80%——>100%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():  # pokemon\bulbasaur\.*png

                images += glob.glob(os.path.join(self.root, name, "*.png"))
                images += glob.glob(os.path.join(self.root, name, "*.jpg"))
                images += glob.glob(os.path.join(self.root, name, "*.jpeg"))
            # 得到1167 'pokemon\\bulbasaur\\000000.png'
            print(len(images), images)
            # todo:打乱一下images
            random.shuffle(images)

            # todo:将文件名，标签分开存储
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                # print('writer into csv file:', filename)

        # todo:读取出来
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        """
        标准化的逆操作
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: imim.open(x).convert('RGB'),  # 将图片转化为RGB三通道数据
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),  # 重新定义一下size
            transforms.RandomRotation(20),  # 随机旋转，这样可以增加数据的多样性
            transforms.CenterCrop(self.resize),  # 上面随机旋转之后会有一些填充，但填充会产生噪声，所以这里用这个方法
            transforms.ToTensor(),  # 将数据转化为tensor数据
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label
def main():
    import visdom
    viz = visdom.Visdom()
    db = SpeechData('dataSet', 64, 'train')
    x, y = next(iter(db))
    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    # todo:使用DataLoader()加载数据,并且在visdom上面显示出来
    loader = DataLoader(db, batch_size=32, shuffle=True)
    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))


if __name__ == '__main__':
    print(1)
    main()