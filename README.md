# RL_BipedalWalker

用 DDPG / TD3 / SAC 求解 `BipedalWalker-v3` 的可复现实现。

> 状态：**骨架搭建中**。本文是基础版说明，记录环境规格、与前置项目的差异、以及已知的三个迁移陷阱。

---

## 这是什么

`BipedalWalker-v3` 是 Gymnasium **Box2D** 套件里的二维双足行走任务：一个只有躯干和两条腿的机器人，
要靠 4 个关节的电机驱动，在不平坦的地形上持续向前行走。

它不是"游戏"，是连续控制领域最经典的基准之一——比 `MountainCarContinuous` 高一个数量级难度，
但比 MuJoCo 套件轻量，适合在一张消费级显卡上跑完整实验。

---

## 环境规格（已在本机实测确认）

| 项 | 值 |
|---|---|
| 环境 ID | `BipedalWalker-v3` |
| 观测维度 | **24**，`Box(24,)` |
| 动作维度 | **4**，`Box(-1, 1, (4,))` |
| 动作含义 | 髋 ×2、膝 ×2 的**电机转速**（不是力矩） |
| 单回合上限 | **1600 步**（`hardcore=True` 为 2000） |
| 达标线 | 平均 **300** 分 |
| 物理引擎 | Box2D 2.3.10 |
| 渲染帧率 | 50 FPS |

### 观测的 24 个分量

躯干角度、躯干角速度、水平速度、垂直速度、
4 个关节的角度与角速度（8 维）、2 个脚的触地标志、**10 条激光测距**。

⚠️ **状态里不含任何绝对坐标**（源码注释原文：`There are no coordinates in the state vector`）。
策略只能靠脚下的地形判断处境，因此必须是平移不变的——学的是"步态"，不是"路线"。

### 奖励结构

```python
shaping = 130 * pos_x / 30 - 5.0 * |躯干角度|
reward  = shaping - prev_shaping          # 势函数塑形：前进 +4.33/米，后退自动为负
reward -= 0.00035 * 80 * |a|              # 每个关节，4 个共约 -0.112/步
if 摔倒 or 后退越过起点:  reward = -100; 终止
```

摔倒 = `-100` **且立即终止**。训练初期几乎每个 episode 都是走两步扑倒，这是主要的探索难点。

---

## 与前置项目 `RL_MountainCarContinuous` 的差异

| | MountainCarContinuous | BipedalWalker-v3 |
|---|---|---|
| 观测 / 动作 | 2 / 1 | **24 / 4** |
| 动力学 | 解析式、光滑 | Box2D，含接触跳变（非光滑） |
| 绝对坐标 | 有 | **没有** |
| 稀疏惩罚 | 到不了就 −100 | **摔倒立即 −100** |
| 样本需求 | ~1 万步 | **100 万步以上** |
| 训练时间 | 1–3 分钟 | **1–3 小时**（单卡） |
| DDPG | 能解 | 大概率失败 |
| 推荐算法 | TD3 ≈ DDPG | **SAC > TD3 > DDPG** |

---

## 三个已知的迁移陷阱

直接复制 `RL_MountainCarContinuous` 的代码并改 `env_id` 会踩这三个坑：

### 1. `max_steps` 会截断 episode

旧项目默认 `max_steps = 999`，而 BipedalWalker 是 1600 步。
应改为从 `env.max_episode_steps` 自动读取，而不是写死。

### 2. 「成功」的判据会反过来（最危险）

旧项目的评估里写的是：

```python
reached = reached or terminated
```

对 MountainCar，`terminated=True` 表示**到顶成功**；
但对 BipedalWalker，`terminated=True` 表示**摔倒失败**（只有极少数是走到终点）。

不改的话，早停会挑选出"摔得最快"的策略，并报告为成功。
必须改成按**回合回报**判定：`return >= 300`。

### 3. 超参数差一个量级

| 参数 | MountainCar | BipedalWalker 建议 |
|---|---|---|
| `reward_scale` | 0.1 | **1.0** |
| `learning_starts` | 1 000 | **25 000** |
| `buffer_size` | 100 000 | **1 000 000** |
| `lr` | 1e-3 | **3e-4** |
| `hidden` | 256 | **400** |
| `eval_every` / `eval_episodes` | 20 / 10 | **100 / 5** |
| `episodes` | 300 | **3000** |

观测归一化在这个环境上更关键（角度 ±π、速度 ±5、触地 0/1、激光 0~1，量级差几十倍），
旧项目的 `RunningMeanStd` 可以直接复用。

---

## 环境安装（已验证可用）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install swig
pip install "gymnasium[box2d]"
```

本机实测：Python 3.13 + **Box2D 2.3.10 有预编译 wheel**，无需现场编译。
（`gymnasium[box2d]` 会顺带安装 `pygame-ce`，它覆盖 `pygame` 包，两者 API 兼容。）

验证：

```powershell
python -c "import gymnasium as gym; e = gym.make('BipedalWalker-v3'); print(e.observation_space, e.action_space)"
```

实时观看：

```powershell
python -c "import gymnasium as gym; e = gym.make('BipedalWalker-v3', render_mode='human'); e.reset(); [e.step(e.action_space.sample()) for _ in range(300)]"
```

---

## 计划结构

计划复用前置项目的骨架，按需扩展：

```
rl/
  config.py        超参数（含 per-env 预设）
  envs.py          环境封装 + 观测归一化 + gym/gymnasium 兼容
  normalizer.py    RunningMeanStd
  networks.py      Actor / Critic / TwinCritic
  buffers.py       ReplayBuffer
  noise.py         高斯噪声
  logger.py        CSV 日志
  utils.py         种子 / 软更新 / checkpoint
  algos/
    base.py        公共逻辑
    ddpg.py
    td3.py
    sac.py         ← 本项目的重点，BipedalWalker 上最稳
train.py
eval.py
tools/
  plot_results.py
  collect_results.py
  bench_device.py
tests/
  test_smoke.py
```

---

## 待办

- [ ] 从 `../RL_MountainCarContinuous` 迁移 `rl/` 骨架
- [ ] 修复「成功判据」与 `max_steps` 两处硬编码
- [ ] 实现 SAC（自动熵调节）
- [ ] 参数按环境区分预设（`--env-id` 自动切换默认值）
- [ ] 跑通 TD3 基线，记录学习曲线
- [ ] 与 SAC 对比，写入 README

---

## 参考

- [BipedalWalker-v3 文档](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
- TD3: [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)
- SAC: [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)
- 前置项目：`../RL_MountainCarContinuous`
