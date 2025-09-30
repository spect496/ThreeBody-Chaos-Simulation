import numpy as np
import matplotlib.pyplot as plt

# 物理常数
G = 1.0  # 在归一化单位中，通常设G=1

# 天体质量（假设三个天体质量相等）
masses = np.array([1.0, 1.0, 1.0])

# 初始条件 - 随机轨道
def initial_conditions():
    """随机生成三体问题的初始条件"""
    # np.random.seed(42)  # 固定随机种子以便复现结果

    # 随机位置 (-1到1之间)
    r1_0 = np.random.uniform(-1, 1, 3)
    r2_0 = np.random.uniform(-1, 1, 3)
    r3_0 = np.random.uniform(-1, 1, 3)

    # 随机速度 (-0.5到0.5之间)
    v1_0 = np.random.uniform(-0.5, 0.5, 3)
    v2_0 = np.random.uniform(-0.5, 0.5, 3)
    v3_0 = np.random.uniform(-0.5, 0.5, 3)

    # 组合初始状态向量
    # 格式: [x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
    y0 = np.concatenate([r1_0, r2_0, r3_0, v1_0, v2_0, v3_0])

    return y0

# 添加扰动
def initial_conditions_perturbed(perturbation=1e-5):
    """在原始初始条件上添加微小扰动"""
    base = initial_conditions()
    # 在第一个天体的x位置添加微小扰动
    perturbed = base.copy()
    perturbed[0] += perturbation  # 扰动第一个天体的x坐标
    return perturbed

# 计算加速度函数
def acceleration(pos, masses):
    """计算天体受到的加速度"""
    acc = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                r_ij = pos[j] - pos[i]
                r = np.linalg.norm(r_ij)
                acc[i] += G * masses[j] * r_ij / r ** 3

    return acc

# 一步rk4
def rk4_step(f, t, y, h, *args):
    """执行一步RK4积分"""
    s1 = f(t, y, *args)
    s2 = f(t + h / 2, y + h / 2 * s1, *args)
    s3 = f(t + h / 2, y + h / 2 * s2, *args)
    s4 = f(t + h, y + h * s3, *args)

    y_new = y + h / 6 * (s1 + 2 * s2 + 2 * s3 + s4)

    return y_new

# 三体问题的微分方程
def three_body_equation(t, y, masses):
    """定义三体问题的微分方程"""
    pos = y[ : 9].reshape(3, 3)
    vel = y[9 : ].reshape(3, 3)

    # 计算加速度
    acc = acceleration(pos, masses)
    dydt = np.concatenate([vel.flatten(), acc.flatten()])

    return dydt

# 可视化函数
def visualize_comparison(traj_orig, traj_pert, perturbation=1e-5):
    """绘制对比图"""
    # 提取每个天体的轨迹
    r1_orig = traj_orig[:, :3]
    r2_orig = traj_orig[:, 3:6]

    r3_orig = traj_orig[:, 6:9]

    r1_pert = traj_pert[:, :3]
    r2_pert = traj_pert[:, 3:6]
    r3_pert = traj_pert[:, 6:9]

    # 创建轨迹图
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 7))

    ax1.plot(r1_orig[:, 0], r1_orig[:, 1], r1_orig[:, 2], label='Body 1')
    ax1.plot(r2_orig[:, 0], r2_orig[:, 1], r2_orig[:, 2], label='Body 2')
    ax1.plot(r3_orig[:, 0], r3_orig[:, 1], r3_orig[:, 2], label='Body 3')
    ax1.set_title('Original Orbit')
    ax1.legend()

    ax2.plot(r1_pert[:, 0], r1_pert[:, 1], r1_pert[:, 2], label='Body 1')
    ax2.plot(r2_pert[:, 0], r2_pert[:, 1], r2_pert[:, 2], label='Body 2')
    ax2.plot(r3_pert[:, 0], r3_pert[:, 1], r3_pert[:, 2], label='Body 3')
    ax2.set_title(f'Perturbed Orbit (Δx₁ = {perturbation:.0e})')

    plt.tight_layout()
    plt.show()

# 模拟函数
def simulate_comparsion(t_max, h, masses, perturbation=1e-5):
    """执行三体问题模拟"""
    y0_orig = initial_conditions()
    y0_pert = initial_conditions_perturbed(perturbation)

    n = int(t_max / h)
    t_values = np.linspace(0, t_max, n+1)

    traj_orig = np.zeros((n+1, len(y0_orig)))
    traj_pert = np.zeros((n+1, len(y0_pert)))

    traj_orig[0] = y0_orig
    traj_pert[0] = y0_pert
    for i in range(1, n+1):
        traj_orig[i] = rk4_step(three_body_equation, t_values[i-1], traj_orig[i-1], h, masses)
        traj_pert[i] = rk4_step(three_body_equation, t_values[i-1],
                                           traj_pert[i-1], h, masses)

    return traj_orig, traj_pert

t_max = 10.0  # 模拟总时间
h = 0.01

traj_orig, traj_pert = simulate_comparsion(t_max, h, masses)
visualize_comparison(traj_orig, traj_pert)