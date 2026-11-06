# Card 5a1: size sweep, test set, environment 4a65fc26

Reference predictors: test MSE minus the toy's floor, against the expected excess (0 for the oracle; offset^2 for the planted offset; (1 - factor)^2 x mean(m(x)^2) for the planted scale), with the standard error of the MSE from the draw's label noise. A difference of more than a few standard errors means the generator, the stated true mean or the floor is wrong.

| variant | toy | n | excess_mse_minus_floor | expected_excess | n_draws | standard_error | difference | difference_in_standard_errors |
|---|---|---|---|---|---|---|---|---|
| oracle | exponential | 2000 | 0.00470152 | 0 | 1 | 0.00359114 | 0.00470152 | 1.3092 |
| oracle | hetero3 | 2000 | -0.00354235 | 0 | 1 | 0.00222548 | -0.00354235 | -1.59172 |
| oracle | heteroscedastic | 2000 | 0.00252912 | 0 | 1 | 0.0224022 | 0.00252912 | 0.112896 |
| oracle | jump__jump=0.0 | 2000 | -0.000768556 | 0 | 1 | 0.00126461 | -0.000768556 | -0.607741 |
| oracle | jump__jump=1.0 | 2000 | -0.000768556 | 0 | 1 | 0.00126461 | -0.000768556 | -0.607741 |
| oracle | ou | 2000 | -0.000106744 | 0 | 1 | 0.000175641 | -0.000106744 | -0.607741 |
| oracle | piecewise | 2000 | 0.000752243 | 0 | 1 | 0.000574582 | 0.000752243 | 1.3092 |
| oracle | toy_a | 2000 | -0.00213488 | 0 | 1 | 0.00351281 | -0.00213488 | -0.607741 |
| oracle | toy_h | 2000 | -0.00554995 | 0 | 1 | 0.00685242 | -0.00554995 | -0.809926 |
| planted_offset | exponential | 2000 | 0.230662 | 0.237258 | 1 | 0.00600821 | -0.00659676 | -1.09796 |
| planted_offset | hetero3 | 2000 | -0.0015465 | 0.00176897 | 1 | 0.00222746 | -0.00331547 | -1.48846 |
| planted_offset | heteroscedastic | 2000 | 0.0454005 | 0.0474809 | 1 | 0.0225975 | -0.00208039 | -0.0920631 |
| planted_offset | jump__jump=0.0 | 2000 | 0.00561501 | 0.00610208 | 1 | 0.00133702 | -0.00048707 | -0.364296 |
| planted_offset | jump__jump=1.0 | 2000 | 0.00213495 | 0.00271572 | 1 | 0.00129481 | -0.000580771 | -0.448537 |
| planted_offset | ou | 2000 | 0.00301258 | 0.00304522 | 1 | 0.000211826 | -3.26366e-05 | -0.154073 |
| planted_offset | piecewise | 2000 | 0.0235939 | 0.0242877 | 1 | 0.000840381 | -0.00069371 | -0.825471 |
| planted_offset | toy_a | 2000 | 0.00619455 | 0.00779905 | 1 | 0.00359987 | -0.0016045 | -0.44571 |
| planted_offset | toy_h | 2000 | 0.00377642 | 0.00836476 | 1 | 0.00689025 | -0.00458834 | -0.665918 |
| planted_scale | exponential | 2000 | 0.358142 | 0.338496 | 1 | 0.0106656 | 0.0196458 | 1.84198 |
| planted_scale | hetero3 | 2000 | -0.00287484 | 0.000973357 | 1 | 0.00223554 | -0.00384819 | -1.72137 |
| planted_scale | heteroscedastic | 2000 | 0.0350366 | 0.0323362 | 1 | 0.0224751 | 0.00270035 | 0.120149 |
| planted_scale | jump__jump=0.0 | 2000 | 0.00413946 | 0.00498032 | 1 | 0.00132081 | -0.000840864 | -0.636625 |
| planted_scale | jump__jump=1.0 | 2000 | 0.00216109 | 0.00315602 | 1 | 0.00131084 | -0.000994925 | -0.758997 |
| planted_scale | ou | 2000 | 0.00279674 | 0.0030021 | 1 | 0.000217916 | -0.000205362 | -0.942395 |
| planted_scale | piecewise | 2000 | 0.0244074 | 0.0227675 | 1 | 0.000895318 | 0.00163988 | 1.83162 |
| planted_scale | toy_a | 2000 | 0.00272493 | 0.00498032 | 1 | 0.00356085 | -0.00225539 | -0.633386 |
| planted_scale | toy_h | 2000 | 0.000510056 | 0.00503051 | 1 | 0.0069165 | -0.00452046 | -0.653576 |

