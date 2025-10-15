# 高阶多项式偏微分方程PINN求解

## 问题描述
求解五阶多项式函数 $u(x,y) = x^5 + y^5 + 3x^3y^2 + 2x^2y^3 - 4xy^4 + 5x - 2y$ 满足的偏微分方程：

$$\Delta u = 26x^3 + 12x^2y - 30xy^2 + 24y^3$$

## 关键推导
1. **控制方程验证**：
   - 二阶导数：
     $$\frac{\partial^2 u}{\partial x^2} = 20x^3 + 18xy^2 + 4y^3$$
     $$\frac{\partial^2 u}{\partial y^2} = 20y^3 + 12x^3 + 12x^2y - 48xy^2$$

2. **高阶约束**：
   - 三阶导数验证（x方向）：
     $$\frac{\partial^3 u}{\partial x^3} = 60x^2 + 18y^2$$
   - 方程两边求导得：
     $$60x^2 + 18y^2 - (78x^2 + 24xy - 30y^2) = 0$$

## PINN实现
### 网络结构
```python
class PoissonPINN(nn.Module):
    def __init__(self, num_layers=4, hidden_size=20):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        # ... 隐藏层构造
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        # 硬边界处理
        return hard_constraint(x, y, net_output)
```

### 创新点
1. **硬边界约束**：通过双线性插值精确满足边界条件
2. **混合验证**：同时计算自动微分和手动解析解
3. **高阶监控**：记录四阶导数差异

### 边界条件处理

本项目采用硬边界约束方法确保神经网络输出在边界上严格满足给定值。以下是边界条件的具体实现：

#### 边界条件函数定义
```python
# 左边界 (x=0)
def boundary_left(y):
    return y**5 - 2*y

# 右边界 (x=1)
def boundary_right(y):
    return 1 + y**5 + 3*y**2 + 2*y**3 - 4*y**4 + 5 - 2*y

# 下边界 (y=0)
def boundary_bottom(x):
    return x**5 + 5*x

# 上边界 (y=1)
def boundary_top(x):
    return x**5 + 1 + 3*x**3 + 2*x**2 - 4*x + 5*x - 2
```

#### 硬边界变换实现
```python
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
```

#### 边界条件处理的优势
1. **严格满足边界条件**：通过硬边界变换，确保神经网络的输出在边界上完全符合给定的边界条件
2. **提高收敛速度**：不需要在损失函数中添加额外的边界条件惩罚项，简化了训练过程
3. **减少训练难度**：专注于学习内部区域的解，降低了神经网络的学习负担
4. **提高解的精度**：特别是在边界附近，解的精度得到显著提升

## 验证方法
### 多阶导数验证
```python
# 四阶导数监控
plt.figure(figsize=(18,4))
plt.subplot(131)
plt.contourf(X, Y, d4u_dx4_diff, levels=50, cmap='viridis')
plt.title('四阶x导数差异')

plt.subplot(132)
plt.contourf(X, Y, d4u_dy4_diff, levels=50, cmap='viridis')
plt.title('四阶y导数差异')

plt.subplot(133)
plt.contourf(X, Y, d4u_dx2y2_diff, levels=50, cmap='viridis')
plt.title('混合四阶导数差异')
plt.savefig('prediction_results/high_order_deriv_diff.png')
```

### 混合验证策略
同时计算自动微分和手动解析解：
```python
# 三阶导数双重验证
d3u_dx3_grad = grad(d2u_dx2, x, torch.ones_like(d2u_dx2), create_graph=True)[0]
d3u_dx3_manual = 60*x**2 + 18*y**2
d3_diff = torch.mean((d3u_dx3_grad - d3u_dx3_manual)**2)
```

## 典型结果
| 指标 | 数值 |
|-------|-------|
| L2误差 | 1.98e-03 |
| 最大误差 | 3.65e-03 |

## 模型对比结果
使用`compare_two_models.py`对OverPINN和PINN模型在y=0.8处的横截面进行了误差对比：

| 模型 | L2误差 | 最大误差 |
|-------|-------|-------|
| OverPINN | 1.916737e-05 | 4.432179e-05 |
| PINN | 3.548821e-05 | 1.191007e-04 |

可视化结果展示了两个模型在y=0.8处的预测值和绝对误差对比，以及误差分布图：

![模型对比可视化结果](model_comparison_results-y=0.8/two_models_comparison.png)

### 误差对比详情

![误差对比详情](model_comparison_results-y=0.8/error_comparison_details.png)

### 结果分析

从对比结果可以看出：
1. OverPINN模型在y=0.8处的横截面表现优于PINN模型，L2误差和最大误差均更小
2. 两个模型在边界附近的误差相对较大，但OverPINN的误差分布更加均匀
3. 误差分布图显示，OverPINN的误差集中区域更小，表明其预测稳定性更好

## 使用说明
```bash
# 训练模型
python 5_Polynomial.py

# 断点续训（示例）
python 5_Polynomial.py --train_from_checkpoint --checkpoint_dir results-2025-08-28-15-15-33
```

![训练损失曲线](results-2025-08-28-15-15-33/training_loss.png)