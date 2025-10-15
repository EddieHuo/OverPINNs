import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 精确解函数
def u_exact(x, y):
    return x**3 + 2*(x**2)*y - y**2 + 5*x

# 边界条件函数
def boundary_left(y):
    return -y**2

def boundary_right(y):
    return -y**2 + 2*y + 6

def boundary_bottom(x):
    return x**3 + 5*x

def boundary_top(x):
    return x**3 + 2*x**2 + 5*x - 1

# 角点值
u_00 = u_exact(0, 0)  # (0,0)
u_01 = u_exact(0, 1)  # (0,1)
u_10 = u_exact(1, 0)  # (1,0)
u_11 = u_exact(1, 1)  # (1,1)

# 硬边界变换函数
def hard_constraint(x, y, net_output):
    """将神经网络输出转换为满足硬边界条件的解"""
    # 双线性插值构造边界条件函数
    A = (1 - x) * boundary_left(y) + x * boundary_right(y) + \
        (1 - y) * boundary_bottom(x) + y * boundary_top(x)
    
    # 减去角点重复计算部分
    B = (1 - x)*(1 - y)*u_00 + (1 - x)*y*u_01 + \
        x*(1 - y)*u_10 + x*y*u_11
    
    # 距离函数 (边界处为0)
    D = x * (1 - x) * y * (1 - y)
    
    return A - B + D * net_output

# 定义神经网络架构
class PoissonPINN(nn.Module):
    def __init__(self, num_layers=4, hidden_size=20):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        net_out = self.net(xy)
        return hard_constraint(x, y, net_out)

# 创建PINN模型实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PoissonPINN().to(device)

# 训练参数
epochs = 10000
lr = 0.0001
batch_size = 128

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=500, verbose=True
)

# 生成训练数据 (内部点)
def generate_internal_points(n):
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    return x, y

# 创建结果保存目录
from datetime import datetime
import os
result_dir = f'results-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
os.makedirs(result_dir, exist_ok=True)

# 训练循环
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 生成内部点
    x, y = generate_internal_points(batch_size)
    
    # 前向传播
    u_pred = model(x, y)
    
    # 计算一阶导数
    du_dx = grad(u_pred, x, torch.ones_like(u_pred), 
                create_graph=True, retain_graph=True)[0]
    du_dy = grad(u_pred, y, torch.ones_like(u_pred), 
                create_graph=True, retain_graph=True)[0]
    
    # 计算二阶导数
    d2u_dx2 = grad(du_dx, x, torch.ones_like(du_dx), 
                  create_graph=True)[0]
    d2u_dx2_manual = 6 * x + 4*y + 5

    d2u_dy2 = grad(du_dy, y, torch.ones_like(du_dy), 
                  create_graph=True)[0]
    d2u_dy2_manual = torch.full_like(d2u_dy2, -2.0)

    # 计算三阶导数
    d3u_dx3 = grad(d2u_dx2, x, torch.ones_like(d2u_dx2), 
                  create_graph=True)[0]         
    d3u_dx3_manual = torch.full_like(d3u_dx3, 6.0)

    d3u_dy3 = grad(d2u_dy2, y, torch.ones_like(d2u_dy2), 
                  create_graph=True)[0]
    d3u_dy3_manual = torch.full_like(d3u_dy3, 0.0)

    # 计算二阶导数的差值 (自动微分 - 手动计算)
    d2u_dx2_diff = d2u_dx2 - d2u_dx2_manual
    d2u_dy2_diff = d2u_dy2 - d2u_dy2_manual

    # 计算三阶导数的差值 (自动微分 - 手动计算)
    d3u_dx3_diff = d3u_dx3 - d3u_dx3_manual
    d3u_dy3_diff = d3u_dy3 - d3u_dy3_manual

    
    # 计算PDE残差
    residual = d2u_dx2 + d2u_dy2 - (6*x + 4*y - 2)
    
    # 计算残差对x和y的一阶偏导（方法1：使用自动微分）
    d_residual_dx_grad = grad(residual, x, torch.ones_like(residual), create_graph=True, retain_graph=True)[0]
    d_residual_dy_grad = grad(residual, y, torch.ones_like(residual), create_graph=True, retain_graph=True)[0]
    
    # 计算残差对x和y的一阶偏导（方法2：手动计算）
    d_residual_dx_manual = d3u_dx3 - 6
    d_residual_dy_manual = d3u_dy3 
    
    # 计算两种方法的差值 (自动微分 - 手动计算)
    d_residual_dx_diff = d_residual_dx_grad - d_residual_dx_manual
    d_residual_dy_diff = d_residual_dy_grad - d_residual_dy_manual 
    
    # 保存偏导结果 (每个epoch保存一次)
    if epoch % 1000 == 0 or epoch == epochs - 1:
        # 只保存一部分样本以减少文件大小
        sample_indices = torch.randint(0, batch_size, (10,))
        # 二阶导数样本
        d2u_dx2_grad_samples = d2u_dx2[sample_indices].detach().cpu().numpy()
        d2u_dx2_manual_samples = d2u_dx2_manual[sample_indices].detach().cpu().numpy()
        d2u_dx2_diff_samples = d2u_dx2_diff[sample_indices].detach().cpu().numpy()

        d2u_dy2_grad_samples = d2u_dy2[sample_indices].detach().cpu().numpy()
        d2u_dy2_manual_samples = d2u_dy2_manual[sample_indices].detach().cpu().numpy()
        d2u_dy2_diff_samples = d2u_dy2_diff[sample_indices].detach().cpu().numpy()

        # 三阶导数样本
        d3u_dx3_grad_samples = d3u_dx3[sample_indices].detach().cpu().numpy()
        d3u_dx3_manual_samples = d3u_dx3_manual[sample_indices].detach().cpu().numpy()
        d3u_dx3_diff_samples = d3u_dx3_diff[sample_indices].detach().cpu().numpy()

        d3u_dy3_grad_samples = d3u_dy3[sample_indices].detach().cpu().numpy()
        d3u_dy3_manual_samples = d3u_dy3_manual[sample_indices].detach().cpu().numpy()
        d3u_dy3_diff_samples = d3u_dy3_diff[sample_indices].detach().cpu().numpy()

        # 残差导数样本
        dx_grad_samples = d_residual_dx_grad[sample_indices].detach().cpu().numpy()
        dx_manual_samples = d_residual_dx_manual[sample_indices].detach().cpu().numpy()
        
        dy_grad_samples = d_residual_dy_grad[sample_indices].detach().cpu().numpy()
        dy_manual_samples = d_residual_dy_manual[sample_indices].detach().cpu().numpy()
        
        # 确保结果目录存在
        os.makedirs(result_dir, exist_ok=True)
        deriv_path = os.path.join(result_dir, f'derivatives_epoch_{epoch}.txt')
        with open(deriv_path, 'w', encoding='utf-8') as f:
            f.write('偏导求解方式比较 (epoch: {})'.format(epoch))
            f.write('\n')
            f.write('二阶导数 d2u_dx2:\n')
            f.write('自动微分  |  手动计算  |  差值 (自动-手动)\n')
            f.write('-' * 40 + '\n')
            for i in range(len(sample_indices)):
                f.write('{:.6f}  |  {:.6f}  |  {:.6f}\n'.format(
                    d2u_dx2_grad_samples[i][0],
                    d2u_dx2_manual_samples[i][0],
                    d2u_dx2_diff_samples[i][0]
                ))
            f.write('\n')
            f.write('二阶导数 d2u_dy2:\n')
            f.write('自动微分  |  手动计算  |  差值 (自动-手动)\n')
            f.write('-' * 40 + '\n')
            for i in range(len(sample_indices)):
                f.write('{:.6f}  |  {:.6f}  |  {:.6f}\n'.format(
                    d2u_dy2_grad_samples[i][0],
                    d2u_dy2_manual_samples[i][0],
                    d2u_dy2_diff_samples[i][0]
                ))
            f.write('\n')
            f.write('三阶导数 d3u_dx3:\n')
            f.write('自动微分  |  手动计算  |  差值 (自动-手动)\n')
            f.write('-' * 40 + '\n')
            for i in range(len(sample_indices)):
                f.write('{:.6f}  |  {:.6f}  |  {:.6f}\n'.format(
                    d3u_dx3_grad_samples[i][0],
                    d3u_dx3_manual_samples[i][0],
                    d3u_dx3_diff_samples[i][0]
                ))
            f.write('\n')
            f.write('三阶导数 d3u_dy3:\n')
            f.write('自动微分  |  手动计算  |  差值 (自动-手动)\n')
            f.write('-' * 40 + '\n')
            for i in range(len(sample_indices)):
                f.write('{:.6f}  |  {:.6f}  |  {:.6f}\n'.format(
                    d3u_dy3_grad_samples[i][0],
                    d3u_dy3_manual_samples[i][0],
                    d3u_dy3_diff_samples[i][0]
                ))
            f.write('\n')
            f.write('残差导数 d_residual_dx:\n')
            f.write('自动微分  |  手动计算  |  差值 (自动-手动)\n')
            f.write('-' * 40 + '\n')
            for i in range(len(sample_indices)):
                dx_diff = dx_grad_samples[i][0] - dx_manual_samples[i][0]
                f.write('{:.6f}  |  {:.6f}  |  {:.6f}\n'.format(
                    dx_grad_samples[i][0],
                    dx_manual_samples[i][0],
                    dx_diff
                ))
            f.write('\n')
            f.write('残差导数 d_residual_dy:\n')
            f.write('自动微分  |  手动计算  |  差值 (自动-手动)\n')
            f.write('-' * 40 + '\n')
            for i in range(len(sample_indices)):
                dy_diff = dy_grad_samples[i][0] - dy_manual_samples[i][0]
                f.write('{:.6f}  |  {:.6f}  |  {:.6f}\n'.format(
                    dy_grad_samples[i][0],
                    dy_manual_samples[i][0],
                    dy_diff
                ))
    
    # 计算各项损失
    residual_loss = torch.mean(residual**2)
    d_residual_dy = d_residual_dy_grad
    d_residual_dx = d_residual_dx_manual
    # d_residual_dy = d_residual_dy_manual
    # derivative_loss = torch.mean(d_residual_dy_diff**2)
    derivative_loss = torch.mean(d_residual_dx**2) + torch.mean(d_residual_dy**2)
    
    loss = residual_loss + derivative_loss
    # loss = residual_loss


    
    # 反向传播
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    
    loss_history.append(loss.item())
    
    if epoch % 1000 == 0 or epoch == epochs - 1:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}')

# 生成测试网格
x_test = np.linspace(0, 1, 100)
y_test = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_test, y_test)
xy_grid = np.vstack([X.ravel(), Y.ravel()]).T

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

print(f'\nSolution Evaluation:')
print(f'L2 Error: {l2_error:.6e}')
print(f'Max Error: {max_error:.6e}')

# 已在训练前创建结果目录

# 保存评估指标到文本文件
metrics_path = os.path.join(result_dir, 'solution_evaluation.txt')
with open(metrics_path, 'w') as f:
    f.write('Solution Evaluation:\n')
    f.write(f'L2 Error: {l2_error:.6e}\n')
    f.write(f'Max Error: {max_error:.6e}\n')

# 可视化结果
plt.figure(figsize=(18, 5))

# 预测解
plt.subplot(1, 3, 1)
plt.contourf(X, Y, u_pred, 50, cmap='jet')
plt.colorbar()
plt.title('PINN Solution')
plt.xlabel('x')
plt.ylabel('y')

# 精确解
plt.subplot(1, 3, 2)
plt.contourf(X, Y, u_true, 50, cmap='jet')
plt.colorbar()
plt.title('Exact Solution')
plt.xlabel('x')
plt.ylabel('y')

# 误差分布
plt.subplot(1, 3, 3)
err_plot = plt.contourf(X, Y, error, 50, cmap='viridis')
plt.colorbar(err_plot)
plt.title('Absolute Error')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'pinn_solution_comparison.png'))
plt.show()

# 损失曲线
plt.figure()
plt.semilogy(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss History')
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'training_loss.png'))
plt.show()

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'pinn_solution_comparison.png'))
plt.show()

# 损失曲线
plt.figure()
plt.semilogy(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss History')
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'training_loss.png'))
plt.show()
