import jax
import jax.numpy as jnp
from jaxpi.problem import PDEProblem
from jaxpi.evaluator import Evaluator
from jaxpi.solver import Solver
import numpy as np
from matplotlib import pyplot as plt

# 定义理论解
def u_exact(x, y):
    return x**3 + 2 * x**2 * y - y**2 + 5 * x

# 定义PDE问题类
class PoissonProblem(PDEProblem):
    def __init__(self, config, key):
        super().__init__(config, key)
        # 定义域边界
        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, 1.0
        
        # 初始化边界条件采样点
        self.init_bc()
        # 初始化内部采样点
        self.init_domain()

    def init_bc(self):
        # 下边界 (y=0)
        self.n_bc_bottom = 100
        x_bottom = jnp.linspace(self.x_min, self.x_max, self.n_bc_bottom)
        y_bottom = jnp.zeros_like(x_bottom)
        u_bottom = u_exact(x_bottom, y_bottom)
        self.bc_bottom = (jnp.stack([x_bottom, y_bottom], axis=1), u_bottom)

        # 上边界 (y=1)
        self.n_bc_top = 100
        x_top = jnp.linspace(self.x_min, self.x_max, self.n_bc_top)
        y_top = jnp.ones_like(x_top)
        u_top = u_exact(x_top, y_top)
        self.bc_top = (jnp.stack([x_top, y_top], axis=1), u_top)

        # 左边界 (x=0)
        self.n_bc_left = 100
        y_left = jnp.linspace(self.y_min, self.y_max, self.n_bc_left)
        x_left = jnp.zeros_like(y_left)
        u_left = u_exact(x_left, y_left)
        self.bc_left = (jnp.stack([x_left, y_left], axis=1), u_left)

        # 右边界 (x=1)
        self.n_bc_right = 100
        y_right = jnp.linspace(self.y_min, self.y_max, self.n_bc_right)
        x_right = jnp.ones_like(y_right)
        u_right = u_exact(x_right, y_right)
        self.bc_right = (jnp.stack([x_right, y_right], axis=1), u_right)

        # 合并所有边界点
        self.bc_points = jnp.concatenate([
            self.bc_bottom[0], self.bc_top[0], self.bc_left[0], self.bc_right[0]
        ], axis=0)
        self.bc_values = jnp.concatenate([
            self.bc_bottom[1], self.bc_top[1], self.bc_left[1], self.bc_right[1]
        ], axis=0)

    def init_domain(self):
        # 内部采样点
        self.n_domain = 10000
        key_x, key_y = jax.random.split(self.key)
        x = jax.random.uniform(key_x, (self.n_domain,), minval=self.x_min, maxval=self.x_max)
        y = jax.random.uniform(key_y, (self.n_domain,), minval=self.y_min, maxval=self.y_max)
        self.domain_points = jnp.stack([x, y], axis=1)

    def pde(self, params, points):
        x, y = points[:, 0], points[:, 1]
        u = self.neural_net(params, points)
        u_x = jax.grad(lambda p: self.neural_net(p, jnp.stack([x, y], axis=1)))(params)
        u_xx = jax.grad(lambda p: jax.vmap(jax.grad(lambda p, x: self.neural_net(p, x)))(p, jnp.stack([x, y], axis=1))[:, 0])(params)
        u_yy = jax.grad(lambda p: jax.vmap(jax.grad(lambda p, x: self.neural_net(p, x)))(p, jnp.stack([x, y], axis=1))[:, 1])(params)
        # 修正PDE计算
        u = jax.vmap(self.neural_net, in_axes=(None, 0))(params, points)
        u_x = jax.vmap(jax.grad(lambda p, x: self.neural_net(p, x)[0]), in_axes=(None, 0))(params, points)[:, 0]
        u_xx = jax.vmap(jax.grad(lambda p, x: jax.grad(lambda p, x: self.neural_net(p, x)[0])(p, x)[0]), in_axes=(None, 0))(params, points)[:, 0]
        u_y = jax.vmap(jax.grad(lambda p, x: self.neural_net(p, x)[0]), in_axes=(None, 0))(params, points)[:, 1]
        u_yy = jax.vmap(jax.grad(lambda p, x: jax.grad(lambda p, x: self.neural_net(p, x)[0])(p, x)[1]), in_axes=(None, 0))(params, points)[:, 1]
        return u_xx + u_yy - (6 * x + 4 * y - 2)

    def loss_fn(self, params):
        # 内部PDE残差损失
        domain_residual = self.pde(params, self.domain_points)
        loss_pde = jnp.mean(domain_residual**2)

        # 边界条件损失
        u_pred = jax.vmap(self.neural_net, in_axes=(None, 0))(params, self.bc_points)
        loss_bc = jnp.mean((u_pred[:, 0] - self.bc_values)**2)

        # 总损失
        return loss_pde + loss_bc

    def compute_errors(self, params):
        # 在均匀网格上计算误差
        n = 100
        x = jnp.linspace(self.x_min, self.x_max, n)
        y = jnp.linspace(self.y_min, self.y_max, n)
        X, Y = jnp.meshgrid(x, y)
        points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

        u_pred = jax.vmap(self.neural_net, in_axes=(None, 0))(params, points)[:, 0]
        u_true = u_exact(points[:, 0], points[:, 1])

        l2_error = jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)
        max_error = jnp.max(jnp.abs(u_pred - u_true))

        return l2_error, max_error

# 配置参数
config = {
    "model": {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_dims": [32, 32, 32],
        "activation": "tanh"
    },
    "solver": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "num_epochs": 10000,
        "print_every": 1000
    }
}

# 主函数
def main():
    key = jax.random.PRNGKey(42)
    problem = PoissonProblem(config, key)
    evaluator = Evaluator(config, problem)
    solver = Solver(config, problem, evaluator)
    params = solver.solve()

    # 计算误差
    l2_error, max_error = problem.compute_errors(params)
    print(f"L2误差: {l2_error:.6e}")
    print(f"最大误差: {max_error:.6e}")

    # 可视化结果
    n = 100
    x = jnp.linspace(0, 1, n)
    y = jnp.linspace(0, 1, n)
    X, Y = jnp.meshgrid(x, y)
    points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    u_pred = jax.vmap(problem.neural_net, in_axes=(None, 0))(params, points)[:, 0].reshape(n, n)
    u_true = u_exact(X, Y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im1 = axes[0].contourf(X, Y, u_true, cmap='viridis')
    axes[0].set_title('理论解')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(X, Y, u_pred, cmap='viridis')
    axes[1].set_title('PINN预测解')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].contourf(X, Y, jnp.abs(u_pred - u_true), cmap='viridis')
    axes[2].set_title('绝对误差')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('poisson_solution.png')
    plt.close()

if __name__ == "__main__":
    main()