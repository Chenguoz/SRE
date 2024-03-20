# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

identity = lambda x: x


class SimpleDataset:
    def __init__(self, data_path, data_file_list, transform, target_transform=identity):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ", "").split(',')
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)
                for j in os.listdir(file_dir):
                    if j.endswith('.jpg'):
                        data.append(file_dir + '/' + j)
                        label.append(k)
                k += 1
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.data[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.label[i] - min(self.label))
        return img, target

    def __len__(self):
        return len(self.label)


class SetDataset:
    def __init__(self, data_path, data_file_list, batch_size, transform, repeat_num=1):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ", "").split(',')
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)
                for j in os.listdir(file_dir):
                    if j.endswith('.jpg'):
                        data.append(file_dir + '/' + j)
                        label.append(k)
                k += 1
        self.data = data
        self.label = label
        self.transform = transform
        self.cl_list = np.unique(self.label).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.data, self.label):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform, repeat_num=repeat_num)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, repeat_num=1):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.repeat_num = repeat_num

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img_replicas = [img] * self.repeat_num
        transformed_imgs = []
        for img_replica in img_replicas:
            if self.transform:
                transformed_img = self.transform(img_replica)
            else:
                transformed_img = img_replica
            transformed_imgs.append(transformed_img)
        img = torch.cat(transformed_imgs, dim=0)

        # img = self.transform(img)

        # target_replicas = [self.cl] * self.N
        # transformed_targets = []
        # for target_replica in target_replicas:
        #     if self.target_transform:
        #         transformed_target = self.target_transform(target_replica)
        #     else:
        #         transformed_target = target_replica
        #     transformed_targets.append(transformed_target)
        # target = torch.cat(transformed_targets, dim=0)
        target = self.target_transform(self.cl)
        # return img, target
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class SimpleDataset_JSON:
    def __init__(self, data_path, data_file, transform, target_transform=identity):
        data = data_path + '/' + data_file
        with open(data, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset_JSON:
    def __init__(self, data_path, data_file, batch_size, transform, repeat_num=1):
        data = data_path + '/' + data_file
        with open(data, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset_JSON(self.sub_meta[cl], cl, transform=transform, repeat_num=repeat_num)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset_JSON:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, repeat_num=1):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

        self.repeat_num = repeat_num

    def __getitem__(self, i):
        # print( '%d -%d' %(self.cl,i))

        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img_replicas = [img] * self.repeat_num
        transformed_imgs = []
        for img_replica in img_replicas:
            if self.transform:
                transformed_img = self.transform(img_replica)
            else:
                transformed_img = img_replica
            transformed_imgs.append(transformed_img)
        img = torch.cat(transformed_imgs, dim=0)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
