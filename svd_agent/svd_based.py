from .svd_agent import SVDAgent
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDAgentAvg(SVDAgent):
    def __init__(self, config):
        super().__init__(config)

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

def svd_based(config):
    return SVDAgentAvg(config)
