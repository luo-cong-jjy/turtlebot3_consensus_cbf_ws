# 三个脚本的 QP / MPC 求解器对比分析

本文面向这三个脚本：

- [mpc_gazebo.py](mpc_gazebo.py)
- [pdnn1.py](pdnn1.py)
- [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py)

目标有两个：

1. 讲清楚三者的 QP / MPC 问题是怎么构建的，求解器是怎么工作的。
2. 给出 [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py) 替换求解器的可执行思路。

## 结论先行

这三个脚本里，真正的差异不在“有没有优化”，而在“优化问题的表达方式”和“求解器范式”完全不同：

- [mpc_gazebo.py](mpc_gazebo.py) 采用 do-mpc 的离散模型 MPC，底层是 CasADi 生成的非线性规划，再交给 IPOPT 做 SQP/内点法求解。
- [pdnn1.py](pdnn1.py) 采用显式构造 QP 矩阵，然后把 KKT 相关的投影约束问题写成一个连续动态系统，再用 Runge-Kutta 积分推进 PDNN 迭代。
- [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py) 本质上是 [pdnn1.py](pdnn1.py) 的升级版：保留 PDNN 求解框架，但把“规划层 QP”和“跟踪层 QP”合并成一个更大的 QP，再用 `gfunction_M` 做限幅映射。

如果你的目标是把第三个脚本的求解器替换成另外两个脚本之一，那么可行路径只有两类：

- 方案 A：把第三个脚本改成 do-mpc / IPOPT 风格，但这不是简单替换求解器，而是要重写成状态空间 MPC。
- 方案 B：保留第三个脚本的 QP 结构，只把 PDNN 迭代部分替换成另一个 QP 求解器，例如 OSQP / qpOASES / CasADi+IPOPT 生成的 NLP 接口。

从“最少改动”角度看，方案 B 更现实；从“算法风格统一”角度看，方案 A 需要重构最大。

## 1. 三个脚本的控制对象和问题类型

### 1.1 [mpc_gazebo.py](mpc_gazebo.py)

这是一个 leader-follower 的视觉/里程计跟踪控制器，但它不是传统显式 QP 求解器，而是 do-mpc 形式的 MPC：

- 状态：`di`、`thetai`
- 控制量：`vc`、`wc`
- 参数：`vL`、`WL`、`alpha`、`beta`
- 动力学：用离散化欧拉法从连续模型得到
- 目标函数：距离误差和角度误差的二次代价
- 约束：控制输入上下界、状态上下界

核心特征是：

- 建模层是符号化的，不是手动组装 QP 矩阵。
- 求解器是 `do_mpc.controller.MPC`，底层调用 CasADi 生成优化问题，再由 IPOPT 求解。
- 这类方法更像“非线性 MPC”，不是你在 [pdnn1.py](pdnn1.py) 里那种手工 QP。

### 1.2 [pdnn1.py](pdnn1.py)

这是一个典型的“手工 QP + 动态系统求解器”写法：

- 先用轨迹/跟踪误差构造 `G`、`W`、`C`、`E`、`b`。
- 再构造 `M = [[W, -E^T], [E, 0]]` 这种增广 KKT 结构。
- 再把约束映射到 `gfuntion()`。
- 最后用 `RungeKutta()` 数值积分 `func()`，让迭代状态 `y` 沿着 PDNN 动力系统收敛。

这意味着它的“求解器”不是一个黑盒优化器，而是一条显式迭代动力学：

$$
\dot y = \gamma \cdot f(y)
$$

然后用 RK4 离散化更新。

### 1.3 [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py)

第三个脚本是最复杂的：它把两层控制融合了。

- 规划层：沿用与 [pdnn1.py](pdnn1.py) 类似的 QP 结构，求解 `u = [ux, uy]`
- 跟踪层：再构建一个独立 QP，求解 `uz = [vc, wc]`
- 合并层：把两个 QP 拼成一个更大的合并 QP，统一求解

因此它不是“一个 QP”，而是“两个 QP 被 block_diag 合并后的超大 QP”。

## 2. QP / MPC 构建方式对比

### 2.1 代价函数形式

#### [mpc_gazebo.py](mpc_gazebo.py)

目标函数是显式的状态误差二次型：

- 距离误差：`(di - des_distance)^2`
- 角度误差：`(thetai - des_angle)^2`
- 控制正则：`set_rterm(vc, wc)`

它的代价是随模型和时变参数进入优化器的，属于标准 MPC 目标。

#### [pdnn1.py](pdnn1.py)

代价函数被折算进 QP 矩阵：

- `Q` 对状态误差加权
- `R` 对控制量加权
- `C = 2 * G^T Q (tildrg + tildrf - des1)` 形成线性项

这里的关键是：代价函数不再以符号表达式存在，而是变成矩阵。

#### [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py)

规划层和跟踪层分别有自己的 `Q / R`，然后用：

- `G_M = block_diag(G, Gz)`
- `Q_M = block_diag(self.Q, self.Qz)`
- `R_M = block_diag(self.R, self.Rz)`

把它们统一成一个大 QP。

这意味着合并脚本已经不是“独立两个问题”了，而是一个耦合后的联合问题。

### 2.2 状态和控制变量

#### [mpc_gazebo.py](mpc_gazebo.py)

- 状态只有 2 维：`di`、`thetai`
- 控制只有 2 维：`vc`、`wc`
- 领导者速度和相对几何参数以时变参数注入

#### [pdnn1.py](pdnn1.py)

- 规划层控制：`u = [ux, uy]`
- 轨迹跟踪层控制：`uz = [vc, wc]`
- 通过 `self.uu` 和 `self.yo` 保存迭代状态

#### [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py)

- 联合向量 `ycv_M` 把规划层和跟踪层叠在一起
- 最后从 `dot_y1_M` 拆出：
  - `self.u`
  - `self.uz`
  - `self.y0x`
  - `self.y0z`

这类结构对“直接替换求解器”不友好，因为它依赖求解器能一次性接受大块矩阵和向量。

### 2.3 约束表达方式

#### [mpc_gazebo.py](mpc_gazebo.py)

约束直接写在 MPC 对象上：

- `mpc.bounds['lower', '_u', 'vc']`
- `mpc.bounds['upper', '_u', 'vc']`
- `mpc.bounds['lower', '_x', 'di']`
- `mpc.bounds['upper', '_x', 'thetai']`

这是“模型层约束”，不是手动拼矩阵。

#### [pdnn1.py](pdnn1.py)

约束通过 `gfuntion()` 做限幅：

- 先定义 `yp`、`ym`
- 再对 `uy[i]` 逐元素裁剪

它是“投影型约束”，本质上是对迭代状态做饱和处理。

#### [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py)

约束被拆成多组矩阵：

- 控制约束：`abarUmin_M`、`abarUmax_M`
- 状态约束：`abarxmin_M`、`abarxmax_M`
- 增量约束：`detabarUmin_M`、`detabarUmax_M`

然后再通过 `gfunction_M()` 统一裁剪。

这类实现比 [pdnn1.py](pdnn1.py) 更容易出形状错，因为不同块的长度并不天然一致。

## 3. 求解器和迭代机制对比

### 3.1 [mpc_gazebo.py](mpc_gazebo.py)：do-mpc + IPOPT

求解链路是：

1. 定义符号模型 `do_mpc.model.Model('discrete')`
2. 定义状态和控制符号
3. 定义连续动力学并离散化
4. `mpc.set_objective()` 定义目标
5. `mpc.bounds[...]` 定义约束
6. `mpc.setup()` 生成优化器
7. 运行时 `u_opt = self.mpc.make_step(x_current)`

优点：

- 易读，结构清晰
- 适合非线性模型和时变参数
- 约束定义自然

缺点：

- 依赖 do-mpc / CasADi / IPOPT
- 每步求解成本通常高于显式 QP
- 不容易直接嵌入你现有的联合 QP 结构

### 3.2 [pdnn1.py](pdnn1.py)：PDNN + RK4

求解链路是：

1. 手工构造 QP 的 Hessian/线性项/约束矩阵
2. 构造投影函数 `gfuntion()`
3. 构造 `func(y)` 作为 PDNN 动力学
4. 用 `RungeKutta()` 对 `func` 做数值积分
5. 从收敛后的状态中提取 `u`

优点：

- 矩阵结构透明
- 和 QP 形式贴得很紧
- 容易把大问题拆成多个块后再拼接

缺点：

- 收敛速度和稳定性依赖步长、增益、初值
- 需要处理很多形状问题和边界裁剪问题
- 实际上是“求解动态系统”，不是显式优化器

### 3.3 [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py)：PDNN 合并求解

求解链路是：

1. 分别构建规划层和跟踪层 QP
2. 用 `block_diag()` 合并成大问题
3. 计算 `W_M / C_M / E_M / b_M / p_M / M_M`
4. 用 `gfunction_M()` 做约束投影
5. 用 RK4 迭代 `ycv_M`
6. 拆出规划层和跟踪层控制量

这说明它保留了 PDNN 范式，但把问题规模和耦合程度放大了。

## 4. 三者的本质差异

### 4.1 [mpc_gazebo.py](mpc_gazebo.py) 是“模型驱动优化”

它关注的是“动态系统 + 目标函数 + 约束”，优化器只是后端。

### 4.2 [pdnn1.py](pdnn1.py) 是“QP 动力系统化”

它关注的是“把 QP 写成可迭代的动力系统”，优化器本身就是一个连续系统。

### 4.3 [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py) 是“多层 QP 的合并求解”

它关注的是“两个 QP 怎么拼成一个更大 QP 并同步收敛”，难点不在单个 QP，而在拼接一致性。

## 5. 第三个脚本替换求解器的可行方案

这里要非常明确：

### 方案 A：把第三个脚本改成 [mpc_gazebo.py](mpc_gazebo.py) 的 do-mpc 求解范式

适用情况：你希望第三个脚本最终只负责一个跟踪任务，且愿意重写控制层。

需要做的事情：

1. 把规划层和跟踪层的大 QP 拆掉。
2. 重新定义状态空间模型，至少要明确状态、控制、时变参数。
3. 将现在的 `G_M / E_M / b_M` 改写为 `do_mpc.model.Model` 中的状态方程。
4. 将约束从 `gfunction_M()` 的裁剪式约束改写为 MPC bounds。
5. 用 `mpc.make_step()` 替代 `RungeKutta(func, y0)`。

风险：

- 工程改动最大。
- 你现在的“规划层 + 跟踪层”结构会被重写。
- 需要重新验证控制性能和稳定性。

结论：这是“算法升级”，不是“求解器替换”。

### 方案 B：把第三个脚本改成 [pdnn1.py](pdnn1.py) 的单层 QP + PDNN 范式

适用情况：你想保留 QP 手工结构，但不想维护两个耦合层。

需要做的事情：

1. 保留规划层或跟踪层中的一个，先把问题降维。
2. 把 `tb3_controller_node_QP_merge.py` 中的大块合并 QP 拆回单层 QP。
3. 复用 `gfunction()` / `RungeKutta()` 这一套结构。
4. 把 `u` 和 `uz` 的职责分开，避免联合求解。

优点：

- 比 do-mpc 更接近原始第三脚本结构。
- 比合并 QP 更容易排查数值问题。

缺点：

- 你会失去“规划层 + 跟踪层统一优化”的优势。

结论：这是“结构简化”，不是“完全等价替换”。

### 方案 C：保留第三个脚本的 QP 结构，把 PDNN 求解器换成显式 QP 求解器

这是我认为最适合你当前目标的方案。

适用情况：

- 你想保留现在的合并 QP 建模方式。
- 你想把求解速度、鲁棒性和可重复性提升上去。
- 你接受改造求解接口，但不想重写建模层。

可选求解器：

- OSQP：适合稀疏凸 QP，实时性强。
- qpOASES：适合在线主动集 QP，控制领域常用。
- CasADi + IPOPT：如果你还想保留更一般的非线性结构。

替换步骤建议：

1. 保留 `G_M / Q_M / R_M / E_M / b_M` 这些矩阵构建。
2. 删除 `gfunction_M()` + RK4 整体迭代求解链。
3. 把当前 QP 直接整理成标准形式：

$$
\min_z \; \frac12 z^T P z + q^T z
$$

约束整理成：

$$
l \le A z \le u
$$

4. 用求解器的标准 API 直接输出 `z_opt`。
5. 从 `z_opt` 中分块提取 `u` 和 `uz`。

这是“最小侵入式替换”，通常也是工程上最稳的路线。

## 6. 对第三脚本的具体替换建议

如果你的最终目标是“把第三个脚本换成另外两个脚本的求解器能力”，我建议按下面顺序评估：

### 6.1 先决定是保留两层结构，还是降成单层

- 如果保留两层结构，优先选显式 QP 求解器，不建议直接换成 do-mpc。
- 如果降成单层结构，才适合直接迁移 [pdnn1.py](pdnn1.py) 的思路。

### 6.2 如果要换成 do-mpc

你需要重写的不是求解器，而是模型层：

- 状态定义
- 时变参数定义
- 目标函数定义
- 状态/控制约束定义
- 控制循环接口

适合“重构项目架构”，不适合“快速替换”。

### 6.3 如果要换成 PDNN 风格

你已经有大部分基础了，真正要做的是：

- 统一维度
- 修正 `gfunction_M()` 的形状处理
- 降低合并 QP 的数值复杂度
- 为 RK4 迭代设置更稳定的初值和步长

适合“保持现在的控制思路，只优化求解稳定性”。

### 6.4 如果要换成 OSQP / qpOASES

这是最推荐的工程方案：

- 你保留第三脚本的 QP 构建。
- 把 PDNN 迭代替换成一个标准 QP solver。
- 这样可直接获得明确的最优解状态、失败状态和迭代信息。

这条路线最符合“第三个脚本的求解器进行替换”这句话的工程含义。

## 7. 推荐的实施顺序

1. 先保留 [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py) 的 QP 构建部分不动。
2. 把当前求解器封装成独立接口，例如 `solve_qp_merge()`。
3. 先接入一个标准 QP 求解器做对照实验。
4. 再决定是否把模型层进一步迁移到 do-mpc。

这样可以避免“一次性重写 + 无法定位问题”的风险。

## 8. 你现在最该注意的点

- [mpc_gazebo.py](mpc_gazebo.py) 不是 QP 迭代求解器，不能直接拿来替代第三脚本的 `gfunction_M()` 逻辑。
- [pdnn1.py](pdnn1.py) 的求解方式可以迁移到第三脚本，但前提是先处理好维度、投影函数和合并块之间的一致性。
- [tb3_controller_node_QP_merge.py](../tb3_controller_node_QP_merge.py) 当前最脆弱的是“合并矩阵维度 + 投影限幅”的组合，不是控制目标本身。

如果你下一步要做代码替换，建议优先走“保留建模层，替换显式 QP 求解器”这条路线。
