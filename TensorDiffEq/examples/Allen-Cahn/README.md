## Discrete Forward Allen-Cahn Equation

Given the non-linear AC equation:

$$u_t - 0.0001u_{xx} + 5u^3 - 5u = 0,$$
$$u(0, x) = x^2 \cos(\pi x),$$
$$u(t,-1) = u(t, 1),$$
$$u_x(t,-1) = u_x(t, 1),$$

with $x \in [-1, 1]$ and $t \in [0, 1]$, we adopt Runge–Kutta methods with q stages. The neural network output is:
$$[u^n_1(x),\dots, u^n_q(x), u^n_{q+1}(x)]$$
where $u^n$ is data at time $t^n$. We extract data from the exact solution at $t_0 = 0.1$ aiming to predict the solution at $t_1 = 0.9$ using a single time-step of $\Delta t = 0.8$.

### Problem Setup

## 🧠 Discrete Forward AC Equation Configuration (SA-PINNs)

| **Component**                            | **Details**                                                                 |
|-----------------------------------------|------------------------------------------------------------------------------|
| **PDE equations**                        | $f^{n+c_j} = 5.0 u^{n+c_j} - 5.0 (u^{n+c_j})^3 + 0.0001 u^{n+c_j}_{xx}$ |
| **Periodic boundary conditions**         | $u(t,-1) = u(t, 1), u_{x}(t,-1) = u_{x}(t, 1)$ |
| **The output of net**                   | $[u^n_1(x),\dots, u^n_q(x), u^n_{q+1}(x)]$ |
| **Layers of net**                        | [2, 128, 128, 128, 128, 1]                                                   |
| **The number of stages (q)**             | 100 (implicitly handled by time stepping strategy)                           |
| **Sample count from collection points at t₀** | 200*                                                                  |
| **Sample count from solutions at t₀**         | 200*                                                                  |
| **t₀ → t₁**                              | 0.1 → 0.9                                                                    |
| **Loss function**                        | $\text{MSE}_0  + \text{MSE}_b + \text{MSE}_c$ |
| **Optimizer**                            | Adam (learning rate = 0.001)                                                 |
| **Adaptive weighting**                   | Enabled (residual + BC weights)                                              |
| **Training iterations**                  | 10,000 TF iterations + 10,000 Newton iterations                              |

\* Same points used for collocation and solutions.

