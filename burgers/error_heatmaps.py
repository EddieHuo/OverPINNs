import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import models
from utils import get_dataset
from jaxpi.utils import restore_checkpoint

# 设置字体支持 - 仅使用Times New Roman
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置学术图表样式
plt.rcParams.update({
    'figure.figsize': (18, 5),
    'axes.labelsize': 24,
    'axes.titlesize': 30,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
# 设置字体支持 - 仅使用Times New Roman
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def load_config(config_path):
    """加载Python配置文件并执行以获取配置"""
    # 动态执行Python配置文件
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()

def plot_error_heatmaps(models_list, u_ref, t_star, x_star, save_dir, l2_errors):
    """绘制包含真实解、OverPINN误差和PINN误差的热力图"""
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")
    
    # 创建一个3列的热力图，调整figsize以容纳下方的误差文本
    fig = plt.figure(figsize=(18, 6))
    
    # 子图1: 准确解
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.pcolor(TT, XX, u_ref, cmap="BuPu")
    plt.colorbar(im1)
    ax1.set_xlabel("t", fontsize=24)
    ax1.set_ylabel("x", fontsize=24)
    ax1.set_title("Exact Solution", fontsize=30)
    ax1.tick_params(axis='both', which='major', labelsize=24)
   
    
    # 子图2: PINN误差
    ax3 = plt.subplot(1, 3, 2)
    u_pred_pinn = models_list[1].u_pred_fn(models_list[1].state.params, t_star, x_star)
    error_pinn = jnp.abs(u_ref - u_pred_pinn)
    im3 = ax3.pcolor(TT, XX, error_pinn, cmap="BuPu")
    plt.colorbar(im3)
    ax3.set_xlabel("t", fontsize=24)
    ax3.set_ylabel("x", fontsize=24)
    ax3.set_title(f"PINN Error", fontsize=30)
    ax3.tick_params(axis='both', which='major', labelsize=24)
    
    # 在PINN误差图下方添加L2误差文本
    ax3.text(0.5, -0.25, f'L2 Error: {l2_errors[1]:.3e}', ha='center', va='top', 
             transform=ax3.transAxes, fontsize=24)
     
    # 子图3: OverPINN误差
    ax2 = plt.subplot(1, 3, 3)
    u_pred_overpinn = models_list[0].u_pred_fn(models_list[0].state.params, t_star, x_star)
    error_overpinn = jnp.abs(u_ref - u_pred_overpinn)
    im2 = ax2.pcolor(TT, XX, error_overpinn, cmap="BuPu")
    plt.colorbar(im2)
    ax2.set_xlabel("t", fontsize=24)
    ax2.set_ylabel("x", fontsize=24)
    ax2.set_title(f"OverPINN Error", fontsize=30)
    ax2.tick_params(axis='both', which='major', labelsize=24)
    
    # 在OverPINN误差图下方添加L2误差文本
    ax2.text(0.5, -0.25, f'L2 Error: {l2_errors[0]:.3e}', ha='center', va='top', 
             transform=ax2.transAxes, fontsize=24)
    # 调整布局以确保文本不会被裁剪
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 调整底部边距
    
    # 保存图像
    fig_path = os.path.join(save_dir, "error_heatmaps_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"Saved error heatmaps comparison to {fig_path}")
    plt.close(fig)

def main(workdir=None):
    # 设置默认工作目录
    if workdir is None:
        workdir = os.path.abspath('.')
    
    # 加载数据集
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0, :]
    
    # 模型目录路径 - PINN和OverPINN目录
    model_dirs = [
        os.path.join(workdir, "OverPINN"),
        os.path.join(workdir, "PINN")
    ]
    
    # 计算并保存L2误差和模型
    l2_errors = []
    models_list = []
    
    for model_dir in model_dirs:
        # 使用项目根目录中的配置文件
        config_path = os.path.join(workdir, "configs", "default.py")
        config = load_config(config_path)
        
        # 检查点路径 - 根据模型类型使用不同的检查点
        if "OverPINN" in model_dir:
            ckpt_path = os.path.join(model_dir, 'ckpt', 'checkpoint_200000')
        else:
            ckpt_path = os.path.join(model_dir, 'ckpt', 'checkpoint_300000')
        
        # 初始化模型
        model = models.Burgers(config, u0, t_star, x_star)
        
        # 加载检查点
        model.state = restore_checkpoint(model.state, ckpt_path)
        models_list.append(model)
        
        # 计算L2误差
        params = model.state.params
        l2_error = model.compute_l2_error(params, u_ref)
        l2_errors.append(l2_error)
        print(f"L2 error for {model_dir}: {l2_error:.3e}")
    
    # 创建保存目录
    compare_dir = os.path.join(workdir, "eval_results", "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    
    # 生成误差热力图对比
    plot_error_heatmaps(models_list, u_ref, t_star, x_star, compare_dir, l2_errors)
    

if __name__ == "__main__":
    main()