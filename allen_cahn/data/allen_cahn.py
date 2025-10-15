import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import expm

# 参数设置
nn = 511  # 空间分辨率
steps = 200  # 时间步数
dom = [-1, 1]  # 空间域范围
t = np.linspace(0, 1, steps + 1)  # 时间步长

# 初始化空间网格
x = np.linspace(dom[0], dom[1], nn + 1)
dx = x[1] - x[0]

# 定义 Allen-Cahn 方程的线性和非线性部分
def linear_part(u):
    # 5u + 0.0001 * u''
    # 使用有限差分法近似二阶导数
    laplacian = np.zeros_like(u)
    laplacian[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
    return 5 * u + 0.0001 * laplacian

def nonlinear_part(u):
    return -5 * u**3

# 初始条件
u = x**2 * np.cos(np.pi * x)

# 时间步长
dt = t[1] - t[0]

# 数值求解
usol = np.zeros((nn + 1, steps + 1))
usol[:, 0] = u

# 使用显式欧拉方法求解
for i in range(steps):
    u = u + dt * (linear_part(u) + nonlinear_part(u))
    usol[:, i + 1] = u

# 绘制空间-时间演化图
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, x, usol, shading='gouraud', cmap='jet')
plt.colorbar(label='u')
plt.xlabel('Time')
plt.ylabel('Space')
plt.title('Allen-Cahn Equation Solution')
plt.tight_layout()

# 保存图片到上级目录的 figures 文件夹
plt.savefig('data/allen_cahn_fd_solution.png', dpi=300, bbox_inches='tight') 
plt.show()

# 保存结果
np.savez('data/allen_cahn.npz', t=t, x=x, usol=usol)