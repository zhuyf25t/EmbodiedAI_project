# SO-101 · MuJoCo · VISTA · ACT

在 MuJoCo 中用 SO-101 完成三个桌面任务：右臂抬起红块（Lift）、右臂把红块叠到绿块上（Stack）、双臂按颜色分拣方块（Sort）。VISTA 看图并调用机械臂工具，采集成功示范。ACT 从示范中学习，再回到同一个环境独立执行。

Lift、Stack 和 Sort 都可以在 MuJoCo 中自检和用 VISTA 采集。Lift 另有经典 ACT 训练和闭环评估入口。

## 从 Lift 开始

以下命令均在 `project/` 下执行。以 Ubuntu 为例，在 Python 虚拟环境中运行 `python -m pip install -r requirements.txt`。无窗口运行 MuJoCo 时设置 `export MUJOCO_GL=egl`。

### 1. 先验证右臂能把红块抓起来

本轮验证环境为 Windows、Python 3.11、MuJoCo 3.3.7。从 `project/` 在 PowerShell 中新建环境可以这样复现：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scene.check_lift --seed 0
```

已激活环境时直接运行：

```bash
python -m scene.check_lift --seed 0
```

脚本在同一初始布局跑两次。第一次让右臂从方块上方接近、闭爪、抬升，第二次让空爪抬到同样高度，红块留在桌面。脚本打印两次的方块高度与场景判定，保存录像。预期结果依次为成功、失败，任一不符就以非零状态退出。

这个脚本可以读取方块初始坐标来校验抓取。每一步都通过正式的关节控制接口执行，方块由夹爪接触带动。这样一段录像和两个判定就能检查机械臂、接触物理和 Lift 裁判。两次运行都属于场景自检。

世界坐标单位为米，桌面为 `z=0`。本轮在 `x∈[0.205,0.218]`、`y∈[-0.010,0.010]` 内均匀采样方块中心，初始方位固定。场景根据方块姿态算出八个顶点中的最低高度，在该高度达到 0.05 m 且两侧指面均与红块有受力接触时判成功。普通 Gym 环境默认在首次成功或达到 120 控制拍时结束。自检继续观察到抬升后的 10 拍保持期。VISTA 采集单独按动作交互轮数结算。

空抓时抓取点沿 y 偏移 6.5 cm，两次末端都以 8.5 cm 为抬升目标。程序检查方块初始位姿相同、正例闭爪时夹住红块且保持期持续成功、空抓全程无接触且红块留在桌面，并比较两次实际末端高度。每次在 `output/check_lift/` 下生成独立目录，含 `grasp.mp4`、`empty.mp4`、同名逐拍 CSV 和 `results.json`。视频底栏使用当前帧对应的 CSV 数值。桌面软接触可能让静止方块的最低点略低于零，CSV 保留带符号的距离。

### 2. 用 VISTA 采一批示范

DeepSeek 官方端点只需在 `project/.env` 填写 `DEEPSEEK_API_KEY`。默认使用 `https://api.deepseek.com` 和 `deepseek-flash`（V4.1 Flash），关闭思考模式。其他兼容端点使用 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY`，需要支持图片和函数工具。首次小批量建议跑 5 局。[DeepSeek 模型说明](https://api-docs.deepseek.com/quick_start/pricing/)

```bash
python run_collect.py --task lift --seed 0 --episodes 5
```

看一局双路实况可用 `python run_collect.py --task lift --seed 0 --episodes 1 --gui`。VISTA 窗口直接读取本批归档的同步图像，左边是腕部 `wrist`，右边是旁观 `front`。下方显示帧号、动作轮数、移动量或夹爪动作，以及模型等待状态。每个控制拍依次播放，模型查图期间停在当前帧。结束后可点击「从头回放」。要重新打开已有双路批次，运行 `python -m scene.live_view data/lift/run04` 即可。

一次命令对应一个任务和一批 episode。这里依次运行 seed 0–4，第一批写入 `data/lift/run01/`，后续批次使用新目录。一次 reset 开始一局。每个通过参数校验并交给控制器的 `play` 算一轮，其中的多个子动作共享一轮。不可达或空抓也占用该轮。局内可以重试；在稳定的子动作边界满足 Lift 判据即成功，达到轮数上限仍未达标则失败。默认每局 20 轮，补采时可用 `--max-play-rounds 24` 改为 24 轮。清单用 `play_rounds` 和 `ticks` 分别记录动作轮数和控制拍数。

**采集成功率 = 成功局数 / 实际开始的局数。** 批次清单记录计划种子、已开始、成功、失败和中断。中断局保留在分母中，尚未开始的种子单列。正式结果使用跑完预定种子的批次，报告成功局数和总局数。课程按任务计分，每项采集成功率须高于 50%。

VISTA 的 `play` 提交一组动作，采集器通过 `control` 执行并回传新图像和动作回执。查图、查历史和读写笔记时，仿真停在当前控制拍，不消耗动作轮数。任务 system prompt 定义方向、抓取窗口和顺序；VISTA 根据两路图像自行选择移动方向和步幅。每局用独立的 `working.md` 记录观察和疑点。物理结束后，VISTA 在终帧复核期间可将有前后帧依据的可复用小技巧写入 `output/vista/lift/guide.md`，再用 `finish` 提交视觉判断。下一局读取共享 guide 和新建的本局笔记。每批另存起止 guide 快照供复盘。

### 3. 用成功示范训练 ACT

在 GPU 主机上，从 `project/` 运行。将本次整理好的 `data/lift/act_seed0_9/` 一并上传，目录内需要有 `manifest.json` 和 10 个 `episodes/*.h5`。先安装与主机驱动兼容的 CUDA 版 PyTorch 2.10.0、torchvision 0.25.0，再安装训练依赖。Linux 无窗口评估需可用的 EGL 驱动。

```bash
python -m pip install -r requirements-act.txt
export MUJOCO_GL=egl
python -u -m act.train --data data/lift/act_seed0_9 --out output/act/lift/run01 --workers 10
```

训练入口从指定批次读取成功且完整的 HDF5，冻结训练清单，并用这份清单计算归一化统计。追加采集后，可在 `--data` 后列出同任务的多个批次，另开一个训练目录。训练日志、归一化统计和权重一起放在 `--out` 下。

Lift 策略读取腕部和旁观 RGB、六维实测关节状态，输出未来 32 拍的关节目标。首次训练时 torchvision 加载 ImageNet ResNet-18 权重，和 Transformer 一起微调。CVAE 后验在训练时读取带 padding mask 的专家动作块；目标是有效动作上的平均 L1 加 10 倍 KL。推理时取潜变量零向量。以 10 Hz 执行每块前 8 拍，再重新观察。训练默认 100 个 epoch、batch size 8；可以用 `--epochs`、`--batch-size` 调整。原命令加 `--resume` 从最近完成的 epoch 继续。三个任务计划共用架构，各自训练权重。

每个 epoch 用固定验证 seed 500–519 独立闭环运行。默认 10 个进程各持有一个 MuJoCo 环境，主进程用一份 GPU 网络批量推理。每个 seed 最多 120 拍；红块最低点离桌至少 50 mm 且两指有受力接触时成功。按 20 局中的成功数选出 `best.pt`，同分保留较早的轮次。`last.pt` 保存续训状态，`epoch_*.json` 保存训练损失和逐局验证结果。训练开始时检查示范种子与验证种子互不重叠。

### 4. 在新布局上独立评估 ACT

```bash
python -u -m evaluation.rollout --checkpoint output/act/lift/run01/best.pt --seed 1000 --episodes 100 --workers 10 --video
```

这条命令从 checkpoint 读取任务、关节顺序和归一化统计，在 seed 1000–1099 上运行 ACT。每次在 `output/eval/lift/` 下另建目录，逐局保存实际关节目标、两指接触及方块净高。`summary.json` 给出成功率、checkpoint 哈希和录像列表。`--video` 按 seed 顺序从实际出现的成功、失败中各取最多两例，回放实际动作，逐拍核对物理判据，再生成腕部和旁观并排的 10 fps 录像。测试种子与训练、选模种子分开。

## Stack 和 Sort

在同一个 seed 做物理正反例，再让 DeepSeek 独立看图操作。以下命令在 `project/` 中运行。

```bash
python -m scene.check_tabletop --task stack --seed 0 --gui
python -m scene.check_tabletop --task sort --seed 0 --gui
python run_collect.py --task stack --seed 0 --episodes 1 --max-play-rounds 24 --gui
python run_collect.py --task sort --seed 0 --episodes 1 --max-play-rounds 24 --gui
```

Stack 自检分别把红块叠到绿块上、放在绿块旁边。Sort 自检分别送达两块、只送达红块。每次自检的录像和逐拍裁判保存在 `output/check_<task>/`。真实采集结束后，可用 `python -m scene.review_tabletop data/sort/run01` 导出指定批次的同步录像、逐个 play 的请求与回执、HDF5 核对结果。把路径换成实际的 Stack 或 Sort 批次即可。

这两个任务的腕相机固定在原腕部镜头位置，采用 88° 垂直视场，光轴朝向张开两指之间；旁观相机固定在桌旁斜上方。Sort 同步录制 `left_wrist`、`right_wrist` 和 `front`。两臂分别使用自己的基座坐标与 TCP；移动与开闭爪须指定 `arm`，另一只手保持原目标。每次复核仍需所有相机和关节在相邻两拍稳定。

本次 seed 0 调试中，Stack 和 Sort 的 `run01` 使用旧 61° 腕相机与有投影的灯光，两项任务从 `run02` 开始使用上面的新配置。首条通过物理判定的示范分别在 `data/stack/run02/`、`data/sort/run03/`，对应录像和工具回执在 `output/vista/<task>/<run>/ep_0000/`。

## 环境、观测和动作

三个任务共用 Gym 风格的 `reset(seed=...)` 和 `step(joint_targets)`。reset 按种子采样初始布局；step 执行**一个控制拍**，返回下一拍观测、奖励、终止标志、超时标志和评估信息。VISTA 的一次工具调用可以跨多个控制拍，ACT 的一个动作块也逐拍走同一个 step。

| 任务 | 用臂与场景 | 环境成功判定 |
| --- | --- | --- |
| Lift | 右臂，规定初始区内一个边长 3 cm 的红块 | 右臂抓住红块并将它抬到离桌面至少 5 cm |
| Stack | 右臂，规定初始区内红、绿两个边长 3 cm 的方块 | 松爪后红块居中叠在绿块上，并保持稳定 |
| Sort | 双臂，红块和绿块从规定的公共初始区出发 | 两块分别进入各自的目标区，并同时保持稳定 |

Lift 按方块最低点测高。Stack 和 Sort 的下列数值是本项目的仿真约定，课程发布正式参数后再对齐。

- 两块边长 30 mm，中心在同一桌面矩形内采样，X 为 193–237 mm，Y 为 −36 至 +36 mm，初始朝向固定。两中心至少有一轴相差 46 mm。当前 X 跨度为 44 mm，因此两块初始沿 Y 分开，上下颜色随机。
- Stack 要求桌面支撑绿块、绿块支撑红块，水平中心距离 ≤8 mm，中心高差为 30±3 mm。
- Sort 的红、绿目标区各为 60×60 mm，中心分别是桌面 (170,−60) 和 (270,+60) mm。两块都须接触桌面，各自所有顶点在桌面的投影均落在同色区域内。
- 两项任务都要求手指已释放，两块的线速 ≤12 mm/s、角速 ≤0.35 rad/s，且全部条件连续满足 5 个 10 Hz 控制拍。

VISTA 能看到具名相机 RGB、任务提示、实测关节状态和本机 TCP。物块真值位姿、接触信息和成功标志供场景裁判与自检使用。VISTA 可以让指定手臂移动末端或开合夹爪，`control/vista.py` 与 `control/skills.py` 将这些操作展开成逐拍关节目标。ACT 直接给出同一格式的关节目标。真机接口与相机标定待课程硬件配置确定后接入。

## 一条示范保存什么

三个任务各有一份 `brains/vista/prompts/<task>.txt`。模型依据当前帧提交 `play`，其中的 `move_by` 采用基座坐标系的毫米增量，起点取当前实测末端，夹爪保持上一次指令。闭爪在两个相邻 10 Hz 观测边界均测得双侧受力接触同一物块且开度变化小于 0.005 时确认夹稳，只向 VISTA 返回 `grasp_confirmed`。试抬后的确认由模型比较新旧 RGB 完成，场景裁判另行结算。最后的视觉复核使用已存帧，期间不再推进物理。

每局 `raw/ep_XXXX/ticks.jsonl` 记录执行前后帧号、本拍动作与离线裁判数据，`frames/` 保存含终帧的原始 PNG。对应的 VISTA 目录中，`calls.jsonl` 是实际模型请求的工具和回执，`actions.jsonl` 给出动作的执行帧区间。一次动作批提前遇到回合终点，只记录已经执行的控制拍。本批的系统提示、起止 guide 快照与模型名分别保存在 VISTA 批次目录和数据清单中。

每个控制拍先读取图像和关节实测值，再下发本拍关节目标。一次“闭爪并抬升”如果占 12 拍，就记录 12 组观测和实际下发的目标。这样 ACT 学到的动作可以直接交给控制接口。

VISTA 的移动和开合爪目标都经过关节插值。每个子动作的最终指令持续保持，直到两个相邻保持帧都到位，且实测关节、开度、腕部与旁观画面均稳定。控制层用容差处理微小抖动。运动拍及保持拍全部录入。一次保持 20 拍仍未稳定时，控制器返回 `SETTLE_TIMEOUT` 并取消同批余下动作。下一轮可用 `observe` 继续持位，若之前是开闭爪也可续发同一指令。稳定后才放行新的移动。已稳定的空抓可重新定位。模型决策时仿真暂停。真机的观测噪声和决策间持位需要另行实测。

从双相机批次开始，`frames/wrist/`、`frames/front/` 里同名 PNG 来自同一物理状态。`calls.jsonl` 中每条 `control_boundary` 的 `stable_frames` 给出可以回看的两帧。归档的初始图、每批回传以及 HDF5 均使用相同的 camera 名称；VISTA 查局部图或像素时显式指定 camera。旧批次的单路图片保存在原目录。

| HDF5 字段 | 类型与形状 | 含义 |
| --- | --- | --- |
| `/observations/images/wrist` | `uint8 [T,H,W,3]` | 执行动作前的腕部 RGB |
| `/observations/images/left_wrist`、`right_wrist` | 各 `uint8 [T,H,W,3]` | Sort 的左腕、右腕 RGB，分别保存 |
| `/observations/images/front` | `uint8 [T,H,W,3]` | 同一时刻的固定旁观 RGB |
| `/observations/state` | `float32 [T,D]` | 同一时刻的关节实测值 |
| `/action` | `float32 [T,D]` | 随后实际下发的关节目标 |

单臂 `D=6`，依次为肩部旋转、肩部俯仰、肘、腕部俯仰、腕部旋转、夹爪。前五维用弧度，夹爪用开度（0 为闭合，1 为全开）。双臂 `D=12`，先左后右。每局的任务提示、seed、控制频率、关节字段和单位随 HDF5 保存。训练时用当前观测监督后续动作块，轨迹末尾用 padding 和 mask 处理。

```text
data/lift/run01/
├── manifest.json          # 本批所有尝试、各局结果及成功示范路径
├── raw/                   # 按 seed 保存原始记录，含失败与中断局
└── episodes/              # 每条完整成功示范一个 HDF5

output/vista/lift/
├── guide.md               # 可跨采集批次更新的任务经验
└── run01/                 # 起始 guide 快照、各局笔记与回执
```

## 代码放在哪里

```text
project/
├── contracts.py           # 单拍观测与具名关节目标，两端共用的字段与单位
├── control/
│   ├── interface.py       # 读取机器人观测、下发目标关节位置
│   ├── mujoco.py          # SO-101 仿真执行器与传感器
│   ├── vista.py           # 逐动作交接、全视角稳定和夹稳确认
│   └── skills.py          # 将末端移动和夹爪操作展开成逐拍目标
├── scene/
│   ├── assets/            # SO-101 模型、桌面、相机与方块
│   ├── env.py             # 共享的 reset / step、仿真时钟与超时处理
│   ├── lift.py            # Lift 的初始布局和成功判定
│   ├── stack.py           # Stack 的初始布局和成功判定
│   ├── sort.py            # Sort 的双臂布局、目标区和成功判定
│   ├── check_lift.py      # 固定 seed 的抓起与空抓自检
│   └── check_tabletop.py  # Stack、Sort 的实物理正反例与录像
├── brains/vista/          # 模型与工具循环、历史帧、工作笔记、跨局指南
├── run_collect.py         # 选择任务，启动一批 VISTA 采集
├── collection/            # 调度回合、记录每拍、结算结果、导出 HDF5
├── act/                   # 数据读取、模型、训练与动作块推理
├── evaluation/            # 闭环执行、成功率汇总与录像，供选模和测试共用
├── data/                  # 按任务、批次存放采集数据
├── output/                # VISTA 经验、ACT 实验与评估产物
└── tmp/MuJoCo-VISTA-ACT/  # 旧项目的参考副本
```

换场景时，`scene` 定义物体怎么摆、何时成功；换机器人后端时，`control` 对齐相机和关节。VISTA 与 ACT 各自消费公开观测，产生的动作最终都经同一条关节执行路径。记录器只依赖单拍数据和回合结果。复用的旧代码放进对应模块，项目运行独立于 `tmp/`。
