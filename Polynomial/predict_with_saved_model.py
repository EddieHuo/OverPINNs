import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义精确解函数
def u_exact(x, y):
    return x**5 + y**5 + 3*x**3*y**2 + 2*x**2*y**3 - 4*x*y**4 + 5*x - 2*y

# 定义边界条件函数和硬约束变换
def boundary_left(y):
    return y**5 - 2*y

def boundary_right(y):
    return 1 + y**5 + 3*y**2 + 2*y**3 - 4*y**4 + 5 - 2*y

def boundary_bottom(x):
    return x**5 + 5*x

def boundary_top(x):
    return x**5 + 1 + 3*x**3 + 2*x**2 - 4*x + 5*x - 2

# 角点值
u_00 = u_exact(0, 0)
u_01 = u_exact(0, 1)
u_10 = u_exact(1, 0)
u_11 = u_exact(1, 1)

# 硬边界变换函数
def hard_constraint(x, y, net_output):
    A = (1 - x) * boundary_left(y) + x * boundary_right(y) + \
        (1 - y) * boundary_bottom(x) + y * boundary_top(x)
    B = (1 - x)*(1 - y)*u_00 + (1 - x)*y*u_01 + \
        x*(1 - y)*u_10 + x*y*u_11
    D = x * (1 - x) * y * (1 - y)
    return A - B + D * net_output

# 定义神经网络架构
class PoissonPINN(torch.nn.Module):
    def __init__(self, num_layers=4, hidden_size=20):
        super().__init__()
        layers = [torch.nn.Linear(2, hidden_size), torch.nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        net_out = self.net(xy)
        return hard_constraint(x, y, net_out)

# 创建模型实例
model = PoissonPINN().to(device)

# 加载保存的模型权重
model_path = os.path.join('results-2025-08-28-15-15-33', 'pinn_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Successfully loaded model from {model_path}')
else:
    raise FileNotFoundError(f'Model file not found: {model_path}')

# 设置模型为评估模式
model.eval()

# 生成测试网格
x_test = np.linspace(0, 1, 100)
y_test = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_test, y_test)

# 转换为Tensor
x_t = torch.tensor(X.ravel(), dtype=torch.float32).unsqueeze(1).to(device)
y_t = torch.tensor(Y.ravel(), dtype=torch.float32).unsqueeze(1).to(device)

# 模型预测
with torch.no_grad():
    u_pred = model(x_t, y_t).cpu().numpy().reshape(100, 100)

# 计算精确解
u_true = u_exact(X, Y)

# 计算误差
error = np.abs(u_pred - u_true)
l2_error = np.sqrt(np.mean(error**2))
max_error = np.max(error)

print('\nSolution Evaluation:')
print(f'L2 Error: {l2_error:.6e}')
print(f'Max Error: {max_error:.6e}')

# 可视化结果1 - 原始的三图对比
plt.figure(figsize=(15, 6), dpi=300)  # 增加高度，提高DPI分辨率
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 使用学术常用字体
    'font.size': 12,                   # 设置基础字体大小
    'axes.titlesize': 14,              # 设置标题字体大小
    'axes.labelsize': 12,              # 设置坐标轴标签字体大小
    'xtick.labelsize': 10,             # 设置x轴刻度标签字体大小
    'ytick.labelsize': 10,             # 设置y轴刻度标签字体大小
    'axes.linewidth': 1.0,             # 设置坐标轴线条宽度
    'mathtext.fontset': 'cm',          # 设置数学公式字体
})

# 预测解
ax1 = plt.subplot(1, 3, 1)
cf1 = ax1.contourf(X, Y, u_pred, 50, cmap='RdBu_r')  # 使用更专业的颜色映射
cbar1 = plt.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title('PINN Solution', fontweight='bold', pad=10)  # 增加标题间距
ax1.set_xlabel('$x$', fontweight='bold')
ax1.set_ylabel('$y$', fontweight='bold')
ax1.set_aspect('equal')  # 保持横纵坐标比例一致
ax1.tick_params(direction='in', which='both')  # 设置刻度向里

# 精确解
ax2 = plt.subplot(1, 3, 2)
cf2 = ax2.contourf(X, Y, u_true, 50, cmap='RdBu_r')
cbar2 = plt.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title('Exact Solution', fontweight='bold', pad=10)
ax2.set_xlabel('$x$', fontweight='bold')
ax2.set_ylabel('$y$', fontweight='bold')
ax2.set_aspect('equal')
ax2.tick_params(direction='in', which='both')

# 误差分布 - 与预测图使用相同的配色方案
ax3 = plt.subplot(1, 3, 3)
cf3 = ax3.contourf(X, Y, error, 50, cmap='RdBu_r')  # 使用与预测图相同的配色
cbar3 = plt.colorbar(cf3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('Error Magnitude', fontweight='bold')
ax3.set_title('Absolute Error', fontweight='bold', pad=10)
ax3.set_xlabel('$x$', fontweight='bold')
ax3.set_ylabel('$y$', fontweight='bold')
ax3.set_aspect('equal')
ax3.tick_params(direction='in', which='both')

# 添加误差指标到图表 - 调整位置避免与图片重合
plt.figtext(0.5, 0.08, f'L2 Error: {l2_error:.6e}, Max Error: {max_error:.6e}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 调整布局，为标题和底部文本留出更多空间
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# 保存可视化结果
result_dir = 'prediction_results'
os.makedirs(result_dir, exist_ok=True)
plt.savefig(os.path.join(result_dir, 'pinn_solution_comparison_loaded.png'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(result_dir, 'pinn_solution_comparison_loaded.eps'), 
            format='eps', dpi=300, bbox_inches='tight')

# 可视化结果2 - 横截面误差对比
plt.figure(figsize=(15, 10), dpi=300)

# 选择横截面的位置 (中间位置)
mid_idx = len(x_test) // 2
x_mid = x_test[mid_idx]  # x=0.5处的横截面

y_mid = y_test[mid_idx]  # y=0.5处的横截面

# 绘制x方向横截面 (y=0.5)
ax4 = plt.subplot(2, 2, 1)
ax4.plot(x_test, u_pred[mid_idx, :], 'b-', linewidth=2, label='PINN Prediction')
ax4.plot(x_test, u_true[mid_idx, :], 'r--', linewidth=2, label='Exact Solution')
ax4.set_title(f'Cross-section at y={y_mid}', fontweight='bold')
ax4.set_xlabel('$x$', fontweight='bold')
ax4.set_ylabel('$u(x, y_mid)$', fontweight='bold')
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.legend(fontsize=10)
ax4.tick_params(direction='in', which='both')

# 绘制y方向横截面 (x=0.5)
ax5 = plt.subplot(2, 2, 2)
ax5.plot(y_test, u_pred[:, mid_idx], 'b-', linewidth=2, label='PINN Prediction')
ax5.plot(y_test, u_true[:, mid_idx], 'r--', linewidth=2, label='Exact Solution')
ax5.set_title(f'Cross-section at x={x_mid}', fontweight='bold')
ax5.set_xlabel('$y$', fontweight='bold')
ax5.set_ylabel('$u(x_mid, y)$', fontweight='bold')
ax5.grid(True, linestyle='--', alpha=0.7)
ax5.legend(fontsize=10)
ax5.tick_params(direction='in', which='both')

# 绘制x方向横截面误差
ax6 = plt.subplot(2, 2, 3)
ax6.plot(x_test, error[mid_idx, :], 'g-', linewidth=2)
ax6.set_title(f'Absolute Error at y={y_mid}', fontweight='bold')
ax6.set_xlabel('$x$', fontweight='bold')
ax6.set_ylabel('Error Magnitude', fontweight='bold')
ax6.grid(True, linestyle='--', alpha=0.7)
ax6.tick_params(direction='in', which='both')

# 绘制y方向横截面误差
ax7 = plt.subplot(2, 2, 4)
ax7.plot(y_test, error[:, mid_idx], 'g-', linewidth=2)
ax7.set_title(f'Absolute Error at x={x_mid}', fontweight='bold')
ax7.set_xlabel('$y$', fontweight='bold')
ax7.set_ylabel('Error Magnitude', fontweight='bold')
ax7.grid(True, linestyle='--', alpha=0.7)
ax7.tick_params(direction='in', which='both')

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'cross_section_comparison.png'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(result_dir, 'cross_section_comparison.eps'), 
            format='eps', dpi=300, bbox_inches='tight')

# 可视化结果3 - 3D误差曲面图
fig3 = plt.figure(figsize=(12, 10), dpi=300)
ax8 = fig3.add_subplot(111, projection='3d')

# 创建3D曲面图
surf = ax8.plot_surface(X, Y, error, cmap='viridis', linewidth=0, antialiased=True)

# 设置标题和标签
ax8.set_title('3D Error Surface', fontweight='bold', pad=20)
ax8.set_xlabel('$x$', fontweight='bold', labelpad=10)
ax8.set_ylabel('$y$', fontweight='bold', labelpad=10)
ax8.set_zlabel('Error Magnitude', fontweight='bold', labelpad=10)

# 添加颜色条
cbar = fig3.colorbar(surf, ax=ax8, shrink=0.5, aspect=5)
cbar.set_label('Error Magnitude', fontweight='bold')

# 设置视角
ax8.view_init(30, 45)  # 仰角30度，方位角45度

plt.tight_layout()
plt.savefig(os.path.join(result_dir, '3d_error_surface.png'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(result_dir, '3d_error_surface.eps'), 
            format='eps', dpi=300, bbox_inches='tight')

# 可视化结果4 - 误差分布直方图
plt.figure(figsize=(10, 6), dpi=300)

# 创建误差分布直方图
plt.hist(error.flatten(), bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Error Distribution Histogram', fontweight='bold', pad=15)
plt.xlabel('Error Magnitude', fontweight='bold')
plt.ylabel('Probability Density', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(direction='in', which='both')

# 添加统计信息文本
stats_text = f'L2 Error: {l2_error:.6e}\nMax Error: {max_error:.6e}\nMean Error: {np.mean(error):.6e}'
plt.figtext(0.85, 0.75, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'error_histogram.png'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(result_dir, 'error_histogram.eps'), 
            format='eps', dpi=300, bbox_inches='tight')

# 显示所有图形
plt.show()

print(f'Visualization saved to {os.path.join(result_dir, "pinn_solution_comparison_loaded.png")}')
print(f'Cross-section comparison saved to {os.path.join(result_dir, "cross_section_comparison.png")}')
print(f'3D error surface saved to {os.path.join(result_dir, "3d_error_surface.png")}')
print(f'Error histogram saved to {os.path.join(result_dir, "error_histogram.png")}')