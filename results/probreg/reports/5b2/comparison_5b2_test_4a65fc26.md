# Card 5b2: change in noise-free excess against 'probreg', test set, environment 4a65fc26

Cell: mean over seeds of the paired change, as % of the reference's noise-free excess (negative = better), with [min, max] over seeds. Grey: the mean is smaller than the reference's seed-to-seed noise. n/a: no 5a record for that toy and size, so not shaded. (noise from X): no 5a record under this environment, so the margin is borrowed from environment X, the one with the most 5a records there. No variant is picked automatically.

| variant | exponential n=2000 | exponential n=250 | hetero3 n=2000 | hetero3 n=250 | heteroscedastic n=2000 | heteroscedastic n=250 | jump__jump=0.0 n=2000 | jump__jump=0.0 n=250 | jump__jump=1.0 n=2000 | jump__jump=1.0 n=250 | ou n=2000 | ou n=250 | piecewise n=2000 | piecewise n=250 | toy_a n=2000 | toy_a n=250 | toy_h n=2000 | toy_h n=250 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classreg | +159.6% [+37.6, +362.8] (noise from 07109356) | +310.8% [+231.4, +397.9] (noise from 07109356) | +70.7% [-0.6, +153.1] (noise from 07109356) | +29.4% [-44.6, +81.7] (noise from 07109356) | +157.2% [+51.1, +272.2] (noise from 07109356) | +135.4% [+96.8, +157.2] (noise from 07109356) | +128.9% [+80.8, +216.9] (noise from 07109356) | +42.5% [+0.7, +73.8] (noise from 07109356) | +117.9% [+77.0, +182.0] (noise from 07109356) | +28.8% [-15.9, +90.0] (noise from 07109356) | +183.6% [+72.9, +319.2] (noise from 07109356) | +139.9% [+51.7, +192.6] (noise from 07109356) | -80.1% [-88.9, -71.8] (noise from 07109356) | -49.1% [-57.7, -40.6] (noise from 07109356) | +130.7% [-4.7, +242.5] (noise from 07109356) | +109.0% [+52.0, +186.4] (noise from 07109356) | +88.2% [+44.4, +136.3] (noise from 07109356) | +54.7% [+21.2, +121.4] (noise from 07109356) |
| classreg_sweep | +159.6% [+37.6, +362.8] (noise from 07109356) | +248.6% [+44.8, +397.9] (noise from 07109356) | +20.6% [-8.7, +58.3] (noise from 07109356) | +30.8% [-40.5, +81.7] (noise from 07109356) | +132.8% [+51.1, +198.9] (noise from 07109356) | +99.6% [-4.3, +206.3] (noise from 07109356) | +128.9% [+80.8, +216.9] (noise from 07109356) | +42.5% [+0.7, +73.8] (noise from 07109356) | +117.9% [+77.0, +182.0] (noise from 07109356) | +32.4% [+0.4, +90.0] (noise from 07109356) | +154.8% [+72.2, +319.2] (noise from 07109356) | +107.3% [+19.5, +230.4] (noise from 07109356) | -83.4% [-88.4, -75.3] (noise from 07109356) | -49.1% [-57.7, -40.6] (noise from 07109356) | +130.7% [-4.7, +242.5] (noise from 07109356) | +109.0% [+52.0, +186.4] (noise from 07109356) | +78.1% [+43.6, +106.2] (noise from 07109356) | +41.7% [+12.6, +91.3] (noise from 07109356) |

| toy | n | reference noise-free excess (mean over seeds) |
|---|---|---|
| exponential | 250 | 0.0592231 |
| exponential | 2000 | 0.0390591 |
| hetero3 | 250 | 0.0127468 |
| hetero3 | 2000 | 0.00157215 |
| heteroscedastic | 250 | 0.11332 |
| heteroscedastic | 2000 | 0.0332725 |
| jump__jump=0.0 | 250 | 0.0139044 |
| jump__jump=0.0 | 2000 | 0.00191773 |
| jump__jump=1.0 | 250 | 0.0209729 |
| jump__jump=1.0 | 2000 | 0.00382527 |
| ou | 250 | 0.00162068 |
| ou | 2000 | 0.000247531 |
| piecewise | 250 | 0.262133 |
| piecewise | 2000 | 0.188252 |
| toy_a | 250 | 0.0199414 |
| toy_a | 2000 | 0.00350765 |
| toy_h | 250 | 0.0337407 |
| toy_h | 2000 | 0.0070504 |
