# Card 5b: change in noise-free excess against 'probreg', test set, environment 4a65fc26

Cell: mean over seeds of the paired change, as % of the reference's noise-free excess (negative = better), with [min, max] over seeds. Grey: the mean is smaller than the reference's seed-to-seed noise. n/a: no 5a record for that toy and size, so not shaded. (noise from X): no 5a record under this environment, so the margin is borrowed from environment X, the one with the most 5a records there. No variant is picked automatically.

| variant | exponential n=2000 | exponential n=250 | hetero3 n=2000 | hetero3 n=250 | heteroscedastic n=2000 | heteroscedastic n=250 | jump__jump=0.0 n=2000 | jump__jump=0.0 n=250 | jump__jump=1.0 n=2000 | jump__jump=1.0 n=250 | ou n=2000 | ou n=250 | piecewise n=2000 | piecewise n=250 | toy_a n=2000 | toy_a n=250 | toy_h n=2000 | toy_h n=250 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm | -60.1% [-66.5, -52.1] (noise from 07109356) | +1100.4% [+418.5, +1500.8] (noise from 07109356) | +199.1% [+87.4, +370.6] (noise from 07109356) | +58.1% [-53.4, +115.2] (noise from 07109356) | +67.7% [+52.5, +90.7] (noise from 07109356) | -17.2% [-23.4, -6.9] (noise from 07109356) | +44.4% [-6.8, +106.6] (noise from 07109356) | +44.3% [+24.3, +68.1] (noise from 07109356) | -12.5% [-31.6, +14.1] (noise from 07109356) | +42.2% [-34.5, +167.8] (noise from 07109356) | +102.6% [+7.9, +271.7] (noise from 07109356) | +73.7% [-5.9, +122.5] (noise from 07109356) | -92.4% [-95.5, -87.3] (noise from 07109356) | -50.4% [-58.8, -42.2] (noise from 07109356) | +99.1% [+6.0, +200.3] (noise from 07109356) | +68.8% [+29.6, +108.1] (noise from 07109356) | +47.5% [+3.4, +85.7] (noise from 07109356) | +79.9% [-18.0, +168.3] (noise from 07109356) |
| linear | +20176.2% [+13431.6, +25577.2] (noise from 07109356) | +16700.6% [+7560.1, +27169.6] (noise from 07109356) | +4053.1% [+3342.2, +4425.7] (noise from 07109356) | +502.7% [+245.1, +697.6] (noise from 07109356) | +6144.9% [+4687.4, +7387.9] (noise from 07109356) | +1765.2% [+1242.6, +2108.5] (noise from 07109356) | +25311.6% [+14958.5, +39002.2] (noise from 07109356) | +3311.6% [+2588.9, +4739.2] (noise from 07109356) | +7836.5% [+5642.7, +11254.6] (noise from 07109356) | +1407.1% [+677.7, +2538.3] (noise from 07109356) | -1.1% [-99.1, +132.9] (noise from 07109356) | +2.5% [-92.4, +71.8] (noise from 07109356) | +52.8% [+0.8, +138.0] (noise from 07109356) | -0.7% [-9.5, +14.8] (noise from 07109356) | +14960.8% [+6988.1, +22028.3] (noise from 07109356) | +2212.4% [+2030.0, +2530.5] (noise from 07109356) | +6276.0% [+4467.6, +7984.5] (noise from 07109356) | +1376.6% [+721.5, +2068.1] (noise from 07109356) |
| oracle | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) | -100.0% [-100.0, -100.0] (noise from 07109356) |
| plain_nn | -86.8% [-96.7, -80.2] (noise from 07109356) | -42.5% [-82.5, +22.6] (noise from 07109356) | -52.5% [-72.9, -28.7] (noise from 07109356) | -53.9% [-81.9, -8.8] (noise from 07109356) | -33.9% [-45.9, -25.3] (noise from 07109356) | +11.0% [-41.2, +54.8] (noise from 07109356) | -19.9% [-46.0, +13.2] (noise from 07109356) | -56.7% [-76.2, -34.4] (noise from 07109356) | -11.4% [-56.9, +26.0] (noise from 07109356) | -8.2% [-46.7, +51.9] (noise from 07109356) | -33.5% [-98.4, +61.3] (noise from 07109356) | -46.7% [-71.6, -23.4] (noise from 07109356) | -97.3% [-98.2, -96.0] (noise from 07109356) | -33.0% [-68.9, -2.5] (noise from 07109356) | +4.4% [-36.0, +62.3] (noise from 07109356) | -26.6% [-48.8, -2.3] (noise from 07109356) | -36.0% [-62.6, -6.2] (noise from 07109356) | +32.6% [-14.6, +100.8] (noise from 07109356) |
| plain_nn_matched | -89.7% [-95.1, -85.5] (noise from 07109356) | -60.9% [-81.3, -46.5] (noise from 07109356) | +3.5% [-78.6, +139.8] (noise from 07109356) | -14.3% [-42.6, +15.3] (noise from 07109356) | -39.7% [-62.0, -26.9] (noise from 07109356) | -25.5% [-78.7, +39.0] (noise from 07109356) | -68.8% [-75.7, -65.1] (noise from 07109356) | -49.3% [-73.3, -31.3] (noise from 07109356) | +14.3% [-11.8, +41.4] (noise from 07109356) | -25.4% [-53.7, +12.5] (noise from 07109356) | -54.5% [-97.1, -17.4] (noise from 07109356) | -43.8% [-89.5, -9.9] (noise from 07109356) | -58.2% [-74.8, -42.1] (noise from 07109356) | -3.8% [-11.9, +9.4] (noise from 07109356) | -35.1% [-60.3, -11.8] (noise from 07109356) | -47.7% [-68.5, -17.3] (noise from 07109356) | -51.1% [-70.3, -37.0] (noise from 07109356) | -15.5% [-38.2, +6.0] (noise from 07109356) |

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
