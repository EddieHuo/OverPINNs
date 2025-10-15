import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset

from ml_collections.config_dict import ConfigDict
import yaml

import datetime

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0, :]
    
    # 修改为同时加载两个检查点
    ckpt_paths = [
        os.path.join(os.path.abspath(workdir), "no_causal_20250421-175416", 'ckpt'),
        os.path.join(os.path.abspath(workdir), "no_causal_20250421-102023", 'ckpt')
    ]
    
    # 生成带有L2误差的图片文件名
    # 在模型加载循环中计算并保存L2误差
    l2_errors = []
    u_preds = []
    for ckpt_path in ckpt_paths:
        model = models.AllenCahn(config, u0, t_star, x_star)
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params  # 新增参数提取
        
        # 计算并保存误差
        l2_error = model.compute_l2_error(params, u_ref)
        l2_errors.append(l2_error)
        u_pred = model.u_pred_fn(params, t_star, x_star)
        u_preds.append(u_pred)
        print(f"L2 error for {ckpt_path}: {l2_error:.3e}")
    
    # 初始化模型并加载检查点
    models_list = []
    for ckpt_path in ckpt_paths:
        model = models.AllenCahn(config, u0, t_star, x_star)
        model.state = restore_checkpoint(model.state, ckpt_path)
        models_list.append(model)

    # 新增误差曲线对比图
    plt.figure(figsize=(10, 6))
    t_index = jnp.where(t_star == 1.0)[0][0]  # 获取t=1.0的索引
    
    for i, model in enumerate(models_list):
        u_pred = model.u_pred_fn(model.state.params, t_star, x_star)
        error = jnp.abs(u_ref[t_index] - u_pred[t_index])
        model_type = "OverPINNs" if "no_causal_20250421-175416" in ckpt_paths[i] else "PINN"
        plt.plot(x_star, error, 
                label=f'{model_type} (L2: {l2_errors[i]:.2e})',
                linestyle='--' if i == 0 else '-',  # 改进版用虚线，原版用实线
                linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Error Comparison at t=1.0')
    plt.legend()
    plt.grid(True)
    
    # 保存对比图
    compare_dir = os.path.join(workdir, "eval_results", "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    plt.savefig(os.path.join(compare_dir, f't1_error_comparison_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'), bbox_inches='tight', dpi=300)

    

    # 使用第一个模型的预测结果生成热力图（根据需要可修改索引）
    error_str = "{:.4e}".format(l2_errors[0]).replace('e-0', 'e-')  # 修正为使用列表中的误差值
    u_pred = u_preds[0]  # 使用保存的预测结果
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")
    
    # plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    # 修改后的绝对误差子图
    plt.subplot(1, 3, 3)
    error_map = plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")  # 添加变量定义
    plt.colorbar(error_map, ax=plt.gca(), location='left')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    # 在右侧创建反转对比图的inset轴
    ax_inset = plt.gca().inset_axes([0.6, 0.6, 0.35, 0.35]) 
    t_index = jnp.where(t_star == 1.0)[0][0]
    ax_inset.plot(x_star[::-1], jnp.abs(u_ref[t_index] - u_pred[t_index]), 
                 'r-', linewidth=2, label='t=1.0 Reversed')
    ax_inset.set_xlabel('x (reversed)')
    ax_inset.set_ylabel('Error')
    ax_inset.legend()
    ax_inset.grid(True)
    plt.tight_layout()

    # Save the figure and config
    save_dir = os.path.join(workdir, "eval_results", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 保存配置文件（修复变量名错误）
    config_path = os.path.join(save_dir, "config.yaml")  # 添加缺失的变量定义
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f)

    # 保存结果图片
    fig_path = os.path.join(save_dir, f"ac_pred_{error_str}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    # fig_path = os.path.join(save_dir, "ac_pred.png")
    # fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    # 新增多时间点对比函数
    def plot_multi_time_errors(models_list, u_ref, t_star, x_star, save_dir):
        # 选择4个代表性时间点（0.25, 0.5, 0.75, 1.0）
        time_points = [0.25, 0.5, 0.75, 1.0]
        plt.figure(figsize=(15, 10))
        
        for idx, t in enumerate(time_points, 1):
            
            plt.subplot(2, 2, idx)
            t_index = jnp.where(t_star >= t)[0][0]  # 获取最近时间索引
            
            for i, model in enumerate(models_list):
                u_pred = model.u_pred_fn(model.state.params, t_star, x_star)
                error = jnp.abs(u_ref[t_index] - u_pred[t_index])
                model_type = "OverPINNs" if i == 0 else "PINN"
                plt.plot(x_star, error, 
                        label=f'{model_type} (L2: {l2_errors[i]:.2e})',
                        linestyle='--' if i == 0 else '-',  # 改进版用虚线，原版用实线
                        linewidth=2)

            # for model in models_list:
            #     u_pred = model.u_pred_fn(model.state.params, t_star, x_star)
            #     error = jnp.abs(u_ref[t_index] - u_pred[t_index])
            #     model_type = "OverPINNs" if "no_causal_20250421-175416" in ckpt_paths[i] else "PINN"
            #     plt.plot(x_star, error, label=f'{model_type} (t={t:.2f})')
            
            plt.xlabel('x')
            plt.ylabel('Absolute Error')
            plt.title(f'Error Comparison at t={t:.2f}')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        fig_path = os.path.join(save_dir, "multi_time_comparison.png")
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"Saved multi-time comparison to {fig_path}")

    # 在现有模型加载后调用新函数
    plot_multi_time_errors(models_list, u_ref, t_star, x_star, compare_dir)

    # 使用第一个模型的预测结果生成热力图（根据需要可修改索引）
    error_str = "{:.4e}".format(l2_errors[0]).replace('e-0', 'e-')  # 修正为使用列表中的误差值
    u_pred = u_preds[0]  # 使用保存的预测结果
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")
    
    # plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    ref_map = plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar(ref_map)

    plt.subplot(1, 3, 2)
    pred_map = plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar(pred_map)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    # Save the figure and config
    save_dir = os.path.join(workdir, "eval_results", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 保存配置文件（修复变量名错误）
    config_path = os.path.join(save_dir, "config.yaml")  # 添加缺失的变量定义
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f)

    # 保存结果图片
    fig_path = os.path.join(save_dir, f"ac_pred_{error_str}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    # 确保所有plt.pcolor调用都赋值给变量
    plt.subplot(1, 3, 1)
    ref_map = plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar(ref_map, ax=plt.gca(), location='left')

    plt.subplot(1, 3, 2)
    pred_map = plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar(pred_map, ax=plt.gca(), location='left')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar(location='left')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    # Save the figure and config
    save_dir = os.path.join(workdir, "eval_results", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 保存配置文件（修复变量名错误）
    config_path = os.path.join(save_dir, "config.yaml")  # 添加缺失的变量定义
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f)

    # 保存结果图片
    fig_path = os.path.join(save_dir, f"ac_pred_{error_str}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
