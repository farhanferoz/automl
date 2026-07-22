"""Single-step probe: same net, same widths, same batch -- kce loss assembly vs w16 loss fn.
If loss values and grads agree at float32-eps scale, the accumulated 6.7e-05 weight drift is
reduction-order noise, not a semantic difference between the trainers."""
import os
import sys

import torch

sys.path.insert(0, "/home/ff235/dev/MLResearch/automl/automl_package/examples")
os.environ.setdefault("AUTOML_DEVICE", "cpu")
import kdropout_converged_width_experiment as kce  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import width_wsel16 as w16  # noqa: E402

REL_FLOAT32_EPS_BUDGET = 1e-5  # generous multiple of float32 eps (1.19e-07) for a 12-term reduction

torch.manual_seed(0)
net = nwn.SharedTrunkPerWidthHeadNet(w_max=12)
x = torch.randn(128, 1)
y = torch.randn(128, 1)
widths = [1, 12, 4, 7]

loss_a = kce._sampled_widths_total_loss(kce.LossType.MSE, net, widths, x, y)
ga = torch.autograd.grad(loss_a, list(net.parameters()), retain_graph=False, allow_unused=True)
loss_fn = w16._make_standard_total_loss_fn(w16.Tier.ONE, torch.ones(128, 1))
loss_b = loss_fn(net, widths, x, y)
gb = torch.autograd.grad(loss_b, list(net.parameters()), retain_graph=False, allow_unused=True)

ld = abs(loss_a.item() - loss_b.item()) / max(abs(loss_a.item()), 1e-12)
gd = max(((a - b).abs().max().item() if (a is not None and b is not None) else (0.0 if a is b else float("inf"))) for a, b in zip(ga, gb))
print(f"loss rel diff = {ld:.3e}   grad max abs diff = {gd:.3e}")
print("VERDICT:", "REDUCTION-ORDER NOISE (same math)" if ld < REL_FLOAT32_EPS_BUDGET and gd < REL_FLOAT32_EPS_BUDGET else "SEMANTIC DIFFERENCE -- investigate")
