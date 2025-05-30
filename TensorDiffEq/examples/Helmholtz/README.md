| Discrete Helmholtz Equation Configuration |  |
|------------------------------------------|--|
| **PDE equation**                         | $u_{xx} + u_{yy} + \lambda u = f(x, y)$ |
| **Domain**                               | $x, y \in [-1, 1]$                      |
| **Boundary conditions**                  | $u(x,y) = 0$ on all domain boundaries   |
| ]**Network output**                      | $u(x, y)$ = $\sin(\pi x)\sin(4\pi y)$   |
| **Layers of net**                        | [2, 50, 50, 50, 50, 1]                  |
| **Initial condition points ($N_0$)**     | 200                                     |
| **Boundary condition points ($N_b$)**    | 100 (distributed across 4 boundaries)   |
| **Collocation points ($N_f$)**           | 100,000                                 |
| **Loss function**                        | MSE (sum of squared residuals + BCs)    |
| **Optimizer**                            | Adam (learning rate = 0.001, β₁ = 0.99) |
| **Adaptive weighting**                   | Enabled (collocation + residual weights)|
| **Training iterations**                  | 10,000 TF iterations + 10,000 L-BFGS    |
| **Test error (relative $L_2$ norm)**     | Computed over a $101 \times 101$ grid   |
