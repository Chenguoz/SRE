import numpy as np
import os
import glob
import argparse
import network.resnet as resnet
import network.resnet12 as resnet12
import network.ResNet12Prior as ResNet12Prior
import torch
import random
import re
import subprocess
import os

model_dict = dict(
    ResNet10=resnet.ResNet10,
    # ResNet12=resnet.ResNet12,
    ResNet12=resnet12.ResNet12,
    ResNet12Prior=ResNet12Prior.ResNet12Prior,
    ResNet18=resnet.ResNet18,
    ResNet34=resnet.ResNet34,
    ResNet34s=resnet.ResNet34s,
    ResNet50=resnet.ResNet50,
    ResNet101=resnet.ResNet101)


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    print(best_file)
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def load_model(model, dir, is_train=False):
    if is_train:
        pretrained_resnet12 = torch.load(dir, map_location='cuda:0')
        layers_to_exclude = ['linear.weight', 'linear.bias', 'linear_rot.weight', 'linear_rot.bias']
        filtered_dict = {k: v for k, v in pretrained_resnet12.items() if k not in layers_to_exclude}
        model.feature.load_state_dict(filtered_dict, strict=False)

    else:
        model_dict = model.state_dict()
        file_dict = torch.load(dir)['state']
        file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
        model_dict.update(file_dict)

        layers_to_exclude = ['feature.linear.weight', 'feature.linear.bias', 'feature.linear_rot.weight', 'feature.linear_rot.bias']
        model_dict = {k: v for k, v in model_dict.items() if k not in layers_to_exclude}
        model.load_state_dict(model_dict, strict=True)

        # model.load_state_dict(model_dict, strict=False)
    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_least_used_gpu():
    """
    Returns the index of the GPU with the lowest usage.
    """
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    gpu_list = output.strip().split('\n')
    gpu_usage = [int(x) for x in gpu_list]
    return gpu_usage.index(min(gpu_usage))


def get_least_used_gpu_memory():
    """
    Returns the index of the GPU with the lowest memory usage.
    """
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    gpu_list = output.strip().split('\n')
    gpu_memory = [int(re.findall(r'\d+', x)[1]) for x in gpu_list]
    return gpu_memory.index(min(gpu_memory))


def create_log_version(filename_prefix):
    major_version = 0
    minor_version = 0
    patch_version = 1

    while True:
        filename = f"{filename_prefix}_{major_version}.{minor_version}.{patch_version}.log"
        if not os.path.exists(filename):
            break
        patch_version += 1
        if patch_version == 10:
            patch_version = 0
            minor_version += 1
        if minor_version == 10:
            minor_version = 0
            major_version += 1

    return filename
