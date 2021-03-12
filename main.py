import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
from dataloaders.datasetGen import SplitGen, PermutedGen
from utils.utils import factory
import random


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Prepare dataloaders
    # train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.train_aug)
    train_dataset, val_dataset = factory(
        'dataloaders', 'base', args.dataset)(args.dataroot, args.train_aug)
    if args.n_permutation > 0:
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                  args.n_permutation,
                                                                                  remap_class=not args.no_class_remap)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)

    # Prepare the Agent (model)
    dataset_name = args.dataset + \
        '_{}'.format(args.first_split_size) + \
        '_{}'.format(args.other_split_size)
    agent_config = {'model_lr': args.model_lr, 'momentum': args.momentum, 'model_weight_decay': args.model_weight_decay,
                    'schedule': args.schedule,
                    'model_type': args.model_type, 'model_name': args.model_name, 'model_weights': args.model_weights,
                    'out_dim': {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space,
                    'model_optimizer': args.model_optimizer,
                    'print_freq': args.print_freq,
                    'gpu': True if args.gpuid[0] >= 0 else False,
                    'with_head': args.with_head,
                    'reset_model_opt': args.reset_model_opt,
                    'reg_coef': args.reg_coef,
                    'head_lr': args.head_lr,
                    'svd_lr': args.svd_lr,
                    'bn_lr': args.bn_lr,
                    'svd_thres': args.svd_thres,
                    'gamma': args.gamma,
                    'dataset_name': dataset_name
                    }

    # agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    agent = factory('svd_agent', args.agent_type,
                    args.agent_name)(agent_config)

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    # task_names = ['2', '1', '3', '4', '5']
    acc_table = OrderedDict()
    acc_table_train = OrderedDict()
    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        task_names = ['All']
        train_dataset_all = torch.utils.data.ConcatDataset(
            train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(
            val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

        acc_table['All'] = {}
        acc_table['All']['All'] = agent.validation(val_loader)

    else:  # Incremental learning
        # Feed data to agent and evaluate agent's performance
        for i in range(len(task_names)):
            train_name = task_names[i]
            print('======================', train_name,
                  '=======================')
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                       batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)

            if args.incremental_class:
                agent.add_valid_output_dim(task_output_space[train_name])

            # Learn
            agent.train_task(train_loader, val_loader)
            torch.cuda.empty_cache()
            # Evaluate
            acc_table[train_name] = OrderedDict()
            acc_table_train[train_name] = OrderedDict()
            for j in range(i + 1):
                val_name = task_names[j]

                print('validation split name:', val_name)
                val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[
                    val_name]
                val_loader = torch.utils.data.DataLoader(val_data,
                                                         batch_size=args.batch_size, shuffle=False,
                                                         num_workers=args.workers)
                acc_table[val_name][train_name] = agent.validation(val_loader)

                print("**************************************************")
                print('training split name:', val_name)
                train_data = train_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[
                    val_name]
                train_loader = torch.utils.data.DataLoader(train_data,
                                                           batch_size=args.batch_size, shuffle=False,
                                                           num_workers=args.workers)
                acc_table_train[val_name][train_name] = agent.validation(
                    train_loader)
                print("**************************************************")

    return acc_table, task_names


def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[1],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='resnet',
                        help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help="The name of actual model for the backbone")

    parser.add_argument('--force_out_dim', type=int, default=0,
                        help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str,
                        default='svd_based', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str,
                        default='svd_based', help="The class name of agent")

    parser.add_argument('--model_optimizer', type=str, default='Adam',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")

    parser.add_argument('--dataroot', type=str, default='../data',
                        help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=0,
                        help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=10)
    parser.add_argument('--other_split_size', type=int, default=10)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")  # class:we need to know specific class,other:no need to know specific class
    parser.add_argument('--train_aug', dest='train_aug', default=True, action='store_false',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=0,
                        help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--model_lr', type=float,
                        default=0.0005, help="Classifier Learning rate")
    parser.add_argument('--head_lr', type=float,
                        default=0.0005, help="Classifier Learning rate")
    parser.add_argument('--svd_lr', type=float, default=0.0005,
                        help="Classifier Learning rate")
    parser.add_argument('--bn_lr', type=float, default=0.0005,
                        help="Classifier Learning rate")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="Learning rate decay")
    parser.add_argument('--svd_thres', type=float,
                        default=1.0, help='reserve eigenvector')

    parser.add_argument('--momentum', type=float, default=0)

    parser.add_argument('--model_weight_decay',
                        type=float, default=1e-5)  # 1e-4

    parser.add_argument('--schedule', nargs="+", type=int, default=[1],
                        help="epoch ")

    parser.add_argument('--print_freq', type=float, default=10,
                        help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")

    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1,
                        help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")

    parser.add_argument('--with_head', dest='with_head', default=False, action='store_true',
                        help="whether constraining head")
    parser.add_argument('--reset_model_opt', dest='reset_model_opt', default=True, action='store_true',
                        help="whether reset optimizer for model at the start of training each tasks")
    parser.add_argument('--reg_coef', type=float, default=100,
                        help="The coefficient for ewc reg")
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    avg_final_acc = np.zeros(args.repeat)
    final_bwt = np.zeros(args.repeat)
    torch.cuda.set_device(args.gpuid[0])
    # Seed
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    for r in range(args.repeat):

        # Run the experiment
        acc_table, task_names = run(args)
        print(acc_table)

        # Calculate average performance across tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * len(task_names)
        bwt_history = [0] * len(task_names)
        for i in range(len(task_names)):
            train_name = task_names[i]
            cls_acc_sum = 0
            backward_transfer = 0
            for j in range(i + 1):
                val_name = task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                backward_transfer += acc_table[val_name][train_name] - \
                    acc_table[val_name][val_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)
            bwt_history[i] = backward_transfer / i if i > 0 else 0
            print('Task', train_name, 'average acc:', avg_acc_history[i])
            print('Task', train_name, 'backward transfer:', bwt_history[i])

        # Gather the final avg accuracy
        avg_final_acc[r] = avg_acc_history[-1]
        final_bwt[r] = bwt_history[-1]

        # Print the summary so far
        print('===Summary of experiment repeats:',
              r + 1, '/', args.repeat, '===')
        print('The last avg acc of all repeats:', avg_final_acc)
        print('The last bwt of all repeats:', final_bwt)
        print('acc mean:', avg_final_acc.mean(),
              'acc std:', avg_final_acc.std())
        print('bwt mean:', final_bwt.mean(), 'bwt std:', final_bwt.std())
