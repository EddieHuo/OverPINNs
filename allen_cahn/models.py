from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class AllenCahn(ForwardIVP):
    def __init__(self, config, u0, t_star, x_star):
        super().__init__(config)

        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, t, x):
        u = self.u_net(params, t, x)
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        u_tx = grad(grad(self.u_net, argnums=1), argnums=2)(params, t, x)
        u_t2x = grad(grad(grad(self.u_net, argnums=1), argnums=2), argnums=2)(params, t, x)
        u_t3x = grad(grad(grad(grad(self.u_net, argnums=1), argnums=2), argnums=2), argnums=2)(params, t, x)
        u_x = grad(self.u_net, argnums=2)(params, t, x)
        u_2x = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        u_3x  = grad(grad(grad(self.u_net, argnums=2), argnums=2), argnums=2)(params, t, x)
        u_4x  = grad(grad(grad(grad(self.u_net, argnums=2), argnums=2), argnums=2), argnums=2)(params, t, x)
        u_5x  = grad(grad(grad(grad(grad(self.u_net, argnums=2), argnums=2), argnums=2), argnums=2), argnums=2)(params, t, x)
        
        # 只保留前两个残差项
        r1 = u_t + 5 * u**3 - 5 * u - 0.0001 * u_2x
        # r1 = 0
        r2 = 0
        # r3 = 0
        r4 = 0
        
        r3 = u_tx + 15 * u**2 * u_x - 5 * u_x - 0.0001 * u_3x

        u_tt = grad(grad(self.u_net, argnums=1), argnums=1)(params, t, x)
        u_xt = grad(grad(self.u_net, argnums=2), argnums=1)(params, t, x)
        u_2xt = grad(grad(grad(self.u_net, argnums=2), argnums=2), argnums=1)(params, t, x)

        # r2 = u_tt + 15 * u**2 * u_t - 5 * u_t - 0.0001 * u_2xt


        # r3 = u_t2x + 30 * u * u_x**2 + 15 * u**2 * u_2x - 5 * u_2x - 0.0001 * u_4x

        # r4 = u_t3x + 30 * u_x**3 + 90 * u * u_x * u_2x + 15 * u**2 * u_3x - 5 * u_3x - 0.0001 * u_5x

        return r1, r2, r3, r4

    def r1_net(self, params, t, x):
        r1, _, _, _ = self.r_net(params, t, x)
        return r1

    def r2_net(self, params, t, x):
        _, r2, _, _ = self.r_net(params, t, x)
        return r2
    def r3_net(self, params, t, x):
        _, _, r3, _ = self.r_net(params, t, x)
        return r3
    def r4_net(self, params, t, x):
        _, _, _, r4 = self.r_net(params, t, x)
        return r4
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        "Compute residuals and weights for causal training"
        # Sort time coordinates
        t_sorted = batch[:, 0].sort()
        r1_pred, r2_pred,  r3_pred, r4_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1]
        )
        
        # Split residuals into chunks
        r1_pred = r_pred.reshape(self.num_chunks, -1)
        r2_pred = r_pred.reshape(self.num_chunks, -1)
        r3_pred = r_pred.reshape(self.num_chunks, -1)
        r4_pred = r_pred.reshape(self.num_chunks, -1)

        r1_l = jnp.mean(r1_pred**2, axis=1)
        r2_l = jnp.mean(r2_pred**2, axis=1)
        r3_l = jnp.mean(r3_pred**2, axis=1)
        r4_l = jnp.mean(r4_pred**2, axis=1)

        r1_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ r1_l)))
        r2_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ r2_l)))
        r3_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ r3_l)))
        r4_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ r4_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([r1_gamma, r2_gamma, r3_gamma, r4_gamma])
        gamma = gamma.min(0)
        # l = jnp.mean(r_pred**2, axis=1)
        # w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return r1_l, r2_l, r3_l, r4_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        # 初始条件预测：在时间 t0 对所有空间点进行并行预测
        # vmap 参数 (None, None, 0) 表示：
        #   params - 保持共享（不向量化）
        #   t0 - 固定时间点（不向量化）
        #   x_star - 沿第0轴向量化（遍历所有空间坐标）
        u_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        
        # 初始条件损失：计算预测值与真实初始条件 u0 的均方误差
        ics_loss = jnp.mean((self.u0 - u_pred) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            r1_l, r2_l, r3_l, r4_l, w = self.res_and_w(params, batch)
            r1_loss = jnp.mean(r1_l * w)
            r2_loss = jnp.mean(r2_l * w)
            r3_loss = jnp.mean(r3_l * w)
            r4_loss = jnp.mean(r4_l * w)
        else:
            r1_pred, r2_pred, r3_pred, r4_pred = self.r_pred_fn( 
                params, batch[:, 0], batch[:, 1]
            )
            r1_loss = jnp.mean(r1_pred**2)
            r2_loss = jnp.mean(r2_pred**2)
            r3_loss = jnp.mean(r3_pred**2)
            r4_loss = jnp.mean(r4_pred**2)

        loss_dict = {
            "ics": ics_loss, 
            "r1": r1_loss,
            "r2": r2_loss,
            "r3": r3_loss,
            "r4": r4_loss,
            }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.u_net, params, self.t0, self.x_star
        )
        

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            # sort the time step for causal loss
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            r1_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r1_net, params, batch[:, 0], batch[:, 1]
            )
            r2_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r2_net, params, batch[:, 0], batch[:, 1]
            )
            r3_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r3_net, params, batch[:, 0], batch[:, 1]
            )
            r4_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r4_net, params, batch[:, 0], batch[:, 1]
            )


            r1_ntk = r1_ntk.reshape(self.num_chunks, -1)
            r2_ntk = r2_ntk.reshape(self.num_chunks, -1)
            r3_ntk = r3_ntk.reshape(self.num_chunks, -1)
            r4_ntk = r4_ntk.reshape(self.num_chunks, -1)

            r1_ntk = jnp.mean(r1_ntk, axis=1)
            r2_ntk = jnp.mean(r2_ntk, axis=1)
            r3_ntk = jnp.mean(r3_ntk, axis=1)
            r4_ntk = jnp.mean(r4_ntk, axis=1)

            _, _, _, _, casual_weights = self.res_and_w(params, batch)
            r1_ntk = r1_ntk * casual_weights
            r2_ntk = r2_ntk * casual_weights
            r3_ntk = r3_ntk * casual_weights
            r4_ntk = r4_ntk * casual_weights            
        else:
            r1_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r1_net, params, batch[:, 0], batch[:, 1]
            )
            r2_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r2_net, params, batch[:, 0], batch[:, 1]
            )
            r3_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r3_net, params, batch[:, 0], batch[:, 1]
            )
            r4_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r4_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {
            "ics": ics_ntk, 
            "r1": r1_ntk,
            "r2": r2_ntk,
            "r3": r3_ntk,
            "r4": r4_ntk,
            }

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class AllenCanhEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.t_star, self.model.x_star)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, _, _, _, causal_weight = self.model.res_and_w(
                state.params, batch
            )
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
