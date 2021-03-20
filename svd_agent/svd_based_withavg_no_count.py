from .agent import Agent
import optim
import torch
import re
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F


class SVDAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.fea_in_hook = {}
        self.fea_in = defaultdict(dict)
        self.fea_in_count = defaultdict(int)

        self.drop_num = 0

        self.regularization_terms = {}
        self.reg_params = {n: p for n,
                           p in self.model.named_parameters() if 'bn' in n}
        self.empFI = False
        self.svd_lr = self.config['model_lr']  # first task
        self.init_model_optimizer()

        self.params_json = {p: n for n, p in self.model.named_parameters()}

    def compute_cov(self, module, fea_in, fea_out):
        if isinstance(module, nn.Linear):
           
            self.update_cov(torch.mean(fea_in[0], 0, True), module.weight)
        

        elif isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding


            fea_in_ = F.unfold(
                torch.mean(fea_in[0], 0, True), kernel_size=kernel_size, padding=padding, stride=stride)

            fea_in_ = fea_in_.permute(0, 2, 1)
            fea_in_ = fea_in_.reshape(-1, fea_in_.shape[-1])
            self.update_cov(fea_in_, module.weight)

        torch.cuda.empty_cache()
        return None

    def update_cov(self, fea_in, k):
        cov = torch.mm(fea_in.transpose(0, 1), fea_in)
        if len(self.fea_in[k]) == 0:
            self.fea_in[k] = cov
        else:
            self.fea_in[k] = self.fea_in[k] + cov

    def init_model_optimizer(self):
        fea_params = [p for n, p in self.model.named_parameters(
        ) if not bool(re.match('last', n)) and 'bn' not in n]
#         cls_params_all = list(p for n,p in self.model.named_children() if bool(re.match('last', n)))[0]
#         cls_params = list(cls_params_all[str(self.task_count+1)].parameters())
        cls_params = [p for n, p in self.model.named_parameters()
                      if bool(re.match('last', n))]
        bn_params = [p for n, p in self.model.named_parameters() if 'bn' in n]
        model_optimizer_arg = {'params': [{'params': fea_params, 'svd': True, 'lr': self.svd_lr,
                                            'thres': self.config['svd_thres']},
                                          {'params': cls_params, 'weight_decay': 0.0,
                                              'lr': self.config['head_lr']},
                                          {'params': bn_params, 'lr': self.config['bn_lr']}],
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad']:
            if self.config['model_optimizer'] == 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(
            optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=self.config['gamma'])

    def train_task(self, train_loader, val_loader=None):
        # 1.Learn the parameters for current task
        self.train_model(train_loader, val_loader)

        if self.reset_model_optimizer:  # Reset model optimizer before learning each task
            self.log('Classifier Optimizer is reset!')
            self.svd_lr = self.config['svd_lr']
            self.init_model_optimizer()
            self.model.zero_grad()
        with torch.no_grad():
            
            self.update_optim_transforms(train_loader)
            

        self.task_count += 1
        if self.reg_params:
            if len(self.regularization_terms) == 0:
                self.regularization_terms = {'importance': defaultdict(
                    list), 'task_param': defaultdict(list)}
            importance = self.calculate_importance(train_loader)
            for n, p in self.reg_params.items():
                self.regularization_terms['importance'][n].append(
                    importance[n].unsqueeze(0))
                self.regularization_terms['task_param'][n].append(
                    p.unsqueeze(0).clone().detach())

    def update_optim_transforms(self, train_loader):
        modules = [m for n, m in self.model.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        handles = []
        for m in modules:
            handles.append(m.register_forward_hook(hook=self.compute_cov))

        
        for i, (inputs, target, task) in enumerate(train_loader):
            if self.config['gpu']:
                inputs = inputs.cuda()
            self.model.forward(inputs)
            
        self.model_optimizer.get_eigens(self.fea_in)
        

        self.model_optimizer.get_transforms()
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

    def calculate_importance(self, dataloader):
        self.log('computing EWC')
        importance = {}
        for n, p in self.reg_params.items():
            importance[n] = p.clone().detach().fill_(0)

        mode = self.model.training
        self.model.eval()
        for _, (inputs, targets, task) in enumerate(dataloader):
            if self.config['gpu']:
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = self.model.forward(inputs)

            if self.empFI:
                ind = targets
            else:
                task_name = task[0] if self.multihead else 'ALL'
                pred = output[task_name] if not isinstance(self.valid_out_dim, int) else output[task_name][:,
                                                                                                           :self.valid_out_dim]
                ind = pred.max(1)[1].flatten()

            loss = self.criterion(output, ind, task, regularization=False)
            self.model.zero_grad()
            loss.backward()

            for n, p in importance.items():
                if self.reg_params[n].grad is not None:
                    p += ((self.reg_params[n].grad ** 2)
                          * len(inputs) / len(dataloader))

        self.model.train(mode=mode)
        return importance

    def reg_loss(self):
        self.reg_step += 1
        reg_loss = 0
        for n, p in self.reg_params.items():
            importance = torch.cat(
                self.regularization_terms['importance'][n], dim=0)
            old_params = torch.cat(
                self.regularization_terms['task_param'][n], dim=0)
            new_params = p.unsqueeze(0).expand(old_params.shape)
            reg_loss += (importance * (new_params - old_params) ** 2).sum()

        self.summarywritter.add_scalar(
            'reg_loss', reg_loss, self.reg_step)
        return reg_loss


def svd_based(config):
    return SVDAgent(config)
