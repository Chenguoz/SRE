import torch.optim
import time
from data.datamgr import SetDataManager
import sys

sys.path.append('./methods')
from methods.protonet import ProtoNet
from methods.priornet import PriorNet
import shutil
from utils import *


def train(params, base_loader, val_loader, model, stop_epoch):
    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    Freezed = True
    for param in model.feature.parameters():
        param.requires_grad = False  # 冻结骨干网络，这部分网络有与训练权重
    if hasattr(model.feature, 'prior'):
        for param in model.feature.prior.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    for epoch in range(0, stop_epoch):
        print('-------------------------------------------')
        print('Start Epoch:', epoch)
        if epoch > 50 and Freezed:
            print('Unfreezed Backbone....')
            Freezed = False
            for param in model.feature.parameters():
                param.requires_grad = True  # 冻结骨干网络，这部分网络有与训练权重


        start = time.time()
        model.train()
        trainObj, top1 = model.train_loop(epoch, base_loader, optimizer)
        # trainObj, top1 = 0,0

        model.eval()
        print('Start Testing Epoch:', epoch)

        valObj, acc = model.test_loop(val_loader, tqdm_bar=epoch == 0)

        if acc > trlog['max_acc']:
            print("best model! save...")
            trlog['max_acc'] = acc
            trlog['max_acc_epoch'] = epoch
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            # if acc > 84:
            #     os.system(
            #         'python test.py --data_path %s --method %s --model_path %s  --test_task_nums 1  --test_n_episode 1000' % (
            #             params.data_path, params.method, outfile))
        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        trlog['train_loss'].append(trainObj)
        trlog['train_acc'].append(top1)
        trlog['val_loss'].append(valObj)
        trlog['val_acc'].append(acc)
        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        lr_scheduler.step()

        print("This epoch use %.2f minutes" % ((time.time() - start) / 60))
        print("train loss is {:.2f}, train acc is {:.2f}".format(trainObj, top1))
        print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
        print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=84, type=int, choices=[84, 224],
                        help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate of the backbone')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 80], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=100, type=int, help='Stopping epoch')
    parser.add_argument('--gpu', default='3', help='gpu id')

    parser.add_argument('--dataset', default='mini_imagenet',
                        choices=['mini_imagenet', 'tiered_imagenet', 'cub', 'cifarfs'])
    parser.add_argument('--data_path', type=str, help='dataset path')

    parser.add_argument('--model', default='ResNet12Prior', choices=['ResNet12', 'ResNet18', 'ResNet12Prior'])
    parser.add_argument('--method', default='priornet',
                        choices=[ 'protonet', 'priornet'])

    parser.add_argument('--train_n_episode', default=600, type=int, help='number of episodes in meta train')
    parser.add_argument('--val_n_episode', default=600, type=int, help='number of episodes in meta val')
    parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
    parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=16, type=int, help='number of unlabeled data in each class')

    parser.add_argument('--extra_dir', default='', help='record additional information')

    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')
    parser.add_argument('--pretrain_path',
                        help='pre-trained model .tar file path')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--reduce_dim', type=int,
                        help='the output dimension of ProtoNet dimensionality reduction layer')

    parser.add_argument('--repeat_num', type=int, default=1,
                        help='')
    parser.add_argument("--ls", type=float, default=1.0,
                        help="RML Lamadas")
    parser.add_argument("--lu", type=float, default=1.0,
                        help="RML Lamadau")
    params = parser.parse_args()
    import platform

    system = platform.system()
    params.gpu = str(get_least_used_gpu_memory())
    print('use gpu:', params.gpu)

    torch.cuda.set_device(get_least_used_gpu_memory())
    if params.model == 'ResNet12' or 'ResNet12Prior':
        params.reduce_dim = 640
    else:
        params.reduce_dim = 512

    if params.n_shot == 1:
        params.train_n_episode = 1000
    else:
        params.train_n_episode = 600

    if params.dataset == 'cifarfs':
        params.image_size = 32

    params.seed = random.randint(1, 999)
    set_seed(params.seed)
    print('seed:', params.seed)

    json_file_read = False
    if params.dataset == 'mini_imagenet':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 64
    elif params.dataset == 'cub':
        base_file = 'base.json'
        val_file = 'val.json'
        json_file_read = True
        params.num_classes = 200
    elif params.dataset == 'tiered_imagenet':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 351
    elif params.dataset == "cifarfs":
        base_file = 'meta-train'
        val_file = 'meta-val'
        params.num_classes = 64

    else:
        ValueError('dataset error')

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot, repeat_num=1)
    base_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
                                  n_episode=params.train_n_episode, json_read=json_file_read, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot, repeat_num=params.repeat_num)
    val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
                                 n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor
    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    params.checkpoint_dir += '_metatrain'
    params.checkpoint_dir += params.extra_dir
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params)
    print(params.checkpoint_dir)
    print(params.pretrain_path)

    if params.method == 'protonet':
        model = ProtoNet(params, model_dict[params.model], **train_few_shot_params)
    elif params.method == 'priornet':
        model = PriorNet(params, model_dict[params.model], **train_few_shot_params)
        shutil.copy('./methods/%s.py' % params.method, str(params.checkpoint_dir))
        shutil.copy('./methods/prior_template.py', str(params.checkpoint_dir))
        shutil.copy('./meta_train.py', str(params.checkpoint_dir))


    model = model.cuda()
    modelfile = os.path.join(params.pretrain_path)
    model = load_model(model, modelfile, is_train=True)

    model = train(params, base_loader, val_loader, model, params.epoch)
    os.system(
        'python test.py --data_path %s --method %s --model_path %s --n_shot %d  --dataset %s --model %s --repeat_num  %d' % (
            params.data_path, params.method,
            os.path.join(
                str(params.checkpoint_dir),
                'best_model.tar'),
            params.n_shot, params.dataset, params.model, params.repeat_num))

