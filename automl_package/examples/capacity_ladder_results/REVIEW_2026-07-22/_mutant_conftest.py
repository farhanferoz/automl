
import sys, os
sys.path.insert(0, os.path.join("/home/ff235/dev/MLResearch/automl", "automl_package", "examples"))
import nested_width_net as nwn
import torch

_orig = nwn.SharedTrunkPerWidthHeadNet.sampled_widths_forward

def _mutant_sampled_widths_forward(self, x, widths):
    h = self.hidden(x)
    means, logvars = [], []
    for k in widths:
        mask = torch.zeros_like(h)
        mask[:, : k - 1] = 1.0  # off-by-one mutant: drops the k-th unit
        mean = self.mean_heads[k - 1](h * mask)
        means.append(mean)
        logvars.append(torch.zeros_like(mean))
    return torch.cat(means, dim=1), torch.cat(logvars, dim=1)

nwn.SharedTrunkPerWidthHeadNet.sampled_widths_forward = _mutant_sampled_widths_forward
