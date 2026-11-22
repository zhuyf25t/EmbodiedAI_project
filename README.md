# SO-101 · MuJoCo · VISTA 示范采集与 ACT 训练

右臂抬红块（Lift）、右臂把红块叠到绿块上（Stack）、双臂分拣（Sort）。三个任务共用一张桌子、两台 SO-101 和一套 10 Hz 控制接口。VISTA 根据相机画面和手臂状态操作，采集器保存每拍的观测与关节目标，并把成功示范导出为 HDF5。ACT 从这些示范学习，每个任务独立训练一份策略。

## 先看三个场景

在 Python 3.11 环境中从仓库根目录运行。Linux 无窗口运行可设置 `MUJOCO_GL=egl`。

```bash
python -m pip install -r requirements.txt
python -m scene.check_tabletop --task lift  --seed 0 --gui
python -m scene.check_tabletop --task stack --seed 0 --gui
python -m scene.check_tabletop --task sort  --seed 0 --gui
python -m scene.check_workspace
```

每次自检用同一 seed 跑一正一反两局，从随机初姿开始显示全部传感器画面，保存到 `output/check_<task>/` 下的新目录。目录包含两段 MP4、每拍实测状态、下发目标、裁判结果和动作回执。脚本可以读取方块位姿做物理自检。VISTA 只能从相机判断方块的位置与朝向。

| 任务 | 初始布局与操作 | 仿真判定与反例 |
| --- | --- | --- |
| Lift | 右臂在右侧作业格抓红块 | 方块最低点离桌 ≥50 mm，且两指对同一红块有受力接触。允许分步抬升。空爪升到同样高度应失败。 |
| Stack | 在右侧作业格采样两块，右臂将红块放在绿块上 | 松爪，桌面支撑绿块、绿块支撑红块；水平中心误差 ≤8 mm、中心高差 30±3 mm。放回绿块旁边应失败。 |
| Sort | 两块始于中央混合区，右臂送红块到图纸右格，左臂送绿块到图纸左格 | 松爪，两块都落在桌面，旋转后的完整投影位于各自目标区。只送达红块应失败。 |

Stack 和 Sort 都要求上述条件以及低线速度、低角速度连续满足 5 个控制拍。容差与保持时间是本项目对课程定性要求的仿真实现。Sort 按 details 的完成态示意图解释方向，俯视实拍图中图纸右格出现在画面左边。

### 桌面、相机和随机化

按 Track 1 details 与参考脚本重建安装布局。以图纸左边为横向原点，三格内侧横向范围分别为 38–204、222–378、396–562 mm，纵向范围为 168–332 mm。两台机械臂同向安装，分别使用自己的基座坐标。局部 +X 远离基座，+Y 指向图纸左侧，+Z 向上。

每个方块边长 30 mm，绕竖轴均匀提议一整圈的角度，再在相应格内提议中心。按实际旋转轮廓检查边界与两块的间距，再检查接近、下降、抬起和送达的逆运动学与机械臂碰撞。路径筛查覆盖桌面、机械臂自身与另一臂，抓取阶段保留真实的爪块接触；相邻方块在操作中仍可能被碰动。最终布局受可达性筛选，位置与朝向的接受分布会偏离原始均匀分布。`output/check_workspace/` 给出每项 16 个 seed 的实际布局及 0°、15°、30°、45° 路径探针，并在同一中央位置做四个角度的真实夹持抬升。

单臂任务录 `wrist`、`front`，Sort 录 `left_wrist`、`right_wrist`、`front`，均为同步 640×480 RGB。腕相机使用参考 URDF 的安装变换和 50° 垂直视场。固定相机使用课程安装位置与 K、D，姿态参照实拍胶带交点校准。校准有效区以外，仿真平滑延伸径向曲线，避免图像边缘出现重复的方块。仿真保留 Menagerie 的黄色臂壳与合成光照，课程参考实拍为白色臂壳。实机的最终外参和关节零点仍须按实际设备校对。

### 从随机初姿开始录制

`env.reset(seed)` 返回随机初姿 `f0`，方块布局和手臂初姿使用独立的随机序列。活动臂从观察姿态附近采样五个关节和开度，排除会碰桌、碰方块或碰另一臂的回位路径。Lift、Stack 的左臂停放，Sort 左右臂依次回位。

1. 采集器先存 `f0`，再通过控制接口插值回位。每拍先存同步 RGB 和实测关节状态，标签是该拍实际下发的关节目标。动作到位后继续发送最终目标，直到连续两拍的关节和各相机画面都稳定。对近焦边缘的亚像素闪动，只在整帧平均通道差 ≤0.15/255 且双向逐像素邻域检查通过时允许一像素容差。所有运动拍和保持拍都录入。
2. 若回位占 H 拍，样本 0 至 H−1 教回位。VISTA 在 `fH` 首次决策，样本 H 教第一拍操作。回位与后续操作同属一局，回位不占 play 次数。模型思考期间仿真时钟暂停。

## 用 VISTA 录制

在仓库根目录把 `.env.example` 复制成 `.env` 并填写密钥。使用 DeepSeek 时填写 `DEEPSEEK_API_KEY`。使用官方 OpenAI SOL 时填写 `OPENAI_BASE_URL=https://api.openai.com/v1`、`OPENAI_MODEL=gpt-6.2-sol`、推理档位 `OPENAI_REASONING_EFFORT` 和 `OPENAI_API_KEY`。SOL 通过 Responses API 接收多相机图像和工具回执。

```bash
# 新建具名批次。三个任务各规划 seed 0–9，共用 10 个 worker
python run_collect.py --out run01 --task lift stack sort --seed 0 --episodes 10 --threads 10 --max-play-rounds 24

# 接着执行已规划、尚未尝试的局
python run_collect.py --resume run01

# 给原有的每个任务追加 10 个 seed，并开始录制
python run_collect.py --resume run01 --add 10 --threads 2

# 执行待录局、补录异常中断的局，并修复缺失的成功导出
python run_collect.py --resume run01 --fix --threads 2

# 在单 worker 模式打开同步相机 GUI
python run_collect.py --out preview --task lift --seed 0 --episodes 1 --gui
```

| 参数 | 约定 |
| --- | --- |
| `--out NAME` / `--resume NAME` | 二选一，只填批次名。`--out` 创建新目录；同名批次用 `--resume` 打开。 |
| `--task`、`--seed`、`--episodes` | 新建时指定任务列表。起始 seed 默认 0，每个任务默认 1 局。共同的 seed 列表写入批次清单。 |
| `--max-play-rounds` | 新建时设定，默认 20。一次 play 可提交多个子动作。回位和模型调用的等待时间均不占 play 轮数。 |
| `--threads` | 所有任务合计的并发上限，默认 1。续录时可以修改。每个并发 worker 独占一套仿真环境。 |
| `--add N` | 从清单已登记的最大 seed 后追加 N 个新 seed，所有任务一起扩容，包括此前已有失败 seed 的任务。 |
| `--fix` | 调度尚未尝试的 seed，以及进程退出或模型异常中断的 seed。每次重跑从同一个 seed 的随机初姿开始，建立独立的 attempt。可与 `--add` 同用。 |

续录继承批次的任务、模型、任务提示和 play 上限。可以在追加示范前调整 `OPENAI_REASONING_EFFORT`；新尝试在 `provider.request_options` 中记录实际设置，旧记录沿用创建批次时的设置。若一次扩容执行到一半退出，用 `--resume NAME --fix` 接续；再次传 `--add N` 表示另外新增 N 个 seed。达到 play 上限属于一次已完成的策略试验，成功或正常失败的 seed 都跳过。每个 seed 的首次实际录制叫 `attempt_01`，手动 `--fix` 重新启动该 seed 才会产生下一个编号。单局内重试模型请求沿用当前 attempt。局末复盘错误保留在已完成的记录中。若物理任务已经成功而 HDF5 导出失败，`--fix` 从该次 raw 重建文件。遇到已有文件与记录不一致时会报错并保留现场。

## 数据、经验和回放

```text
data/run01/
  manifest.json
  lift/ep_0000.h5
  stack/ep_0000.h5
  sort/ep_0000.h5

output/vista/run01/
  lift/
  stack/
  sort/
    system_prompt.txt
    guide.md
    guide_start.md
    guide_end.md
    ep_0000/
      attempt_01/
        actions.jsonl  calls.jsonl  messages.jsonl  working.md  outcome.json
        raw/
          reset.json  ticks.jsonl  front_ruler.png
          frames/{left_wrist,right_wrist,front}/
      attempt_02/
        ...
```

- 一个成功 seed 发布一份 HDF5。T 个动作与 T 份执行前观测按行对齐，字段为 `observations/images/<camera>`、`observations/state` 和 `action`。单臂 6 维，双臂 12 维且左臂在前。每臂前五维是弧度，第六维是 0–1 开度。HDF5 属性包含关节名、单位、相机、版本和回位交接帧。
- 批次清单的 `tasks.<task>.episodes` 按 seed 保存历次尝试。成功记录含批次相对路径和 SHA-256。任务的 `summary` 分别记录规划 seed 数、开始的 seed 数、尝试次数、首试成功数、中断次数和已导出的 seed 数。整批训练数据可随 `data/run01/` 一起复制。
- `raw` 供 VISTA 在线查图、导出和回放使用。T 个动作对应每台相机 T+1 张原图，包含初帧和动作结束后的终帧。`ticks.jsonl` 记录每拍的阶段、观测帧号、实测关节、实际动作和执行后的裁判结果。失败尝试也保留原始记录。
- 每次尝试独立保存 working、工具回执和对话。在线 guide 由同批次、同任务的 worker 共享，修改使用文件锁和版本校验。首次创建批次时初始化该任务的 guide。

复核需要同名批次的 `data` 和 `output/vista` 目录。在仓库根目录运行：

```bash
python -m scene.review_tabletop data/run01/sort --seed 0 --attempt 2 --replay
```

复核工具在所选的 attempt 中生成同步录像 `episode.mp4`、动作回执 `play_receipts.json`、关键帧和 `verification.json`。`--replay` 额外从随机初姿重放动作，对照逐拍状态、裁判和边界帧图像。省略 seed 和 attempt 会复核该任务下所有有记录的已结束尝试。

## 三个任务的实际录制结果

以下是批次 `run01` 的首批 seed 0–9。三个任务都使用 SOL xhigh，每局最多 24 轮 play。

| 任务 | 首试成功 | 补录 | 已保存示范 |
| --- | ---: | ---: | ---: |
| Lift | 10/10 | — | 10 |
| Stack | 10/10 | — | 10 |
| Sort | 3/10 | 7/7 | 10 |

Sort 首轮有 7 局在动作期间收到 HTTP 429，原记录保存在对应 seed 的 `attempt_01`。双并发补录成功的 7 局保存在 `attempt_02`。在 seed 0–9 上，Sort 一共启动 17 次，10 个 seed 最终都有示范。

同一批次的 seed 10–19 使用 SOL high、5 个并发 worker 和相同的 24 轮上限。Lift、Stack 首试各成功 10/10。Sort 首试成功 7/10，seed 14 首次遇到 HTTP 429，补录在第 23 轮成功；seed 16 和 18 用满 24 轮，按策略失败保留。新增示范共 28 条。全批次可训练的 HDF5 为 Lift 20 条、Stack 20 条、Sort 18 条。

## 控制与模型工具

`move_by(dx_mm,dy_mm,dz_mm)` 从当前实测 TCP 起步，每轴至多 100 mm；可选 `grasp_yaw_deg` 是相对基座中性抓取朝向的绝对角，范围 ±45°，省略时保持上一朝向。控制层负责逆运动学和逐拍插值。双臂操作要指定活动臂，另一臂保持当前关节目标。闭爪只返回夹稳与否，抬升后由 VISTA 看图复核。未稳定会收到 `SETTLE_TIMEOUT`，下一次用 `observe` 或续发同一夹爪指令等待。

首轮另附一张带毫米刻度的固定相机副本，供模型目视读取桌上方块的顶面位置。标尺根据相机标定和固定基座绘制。示范保存同步相机的原始 RGB。`read_ruler` 可按模型指定的毫米范围逐点读取同一张 front 原图的 RGB，由模型自己判断颜色、方位和步幅。若下降受阻且实测关节和图像已稳定，允许原朝向纯竖直上退重新对准。已静止而实测仍未到指定位置的移动报 `MOTION_BLOCKED`，整批后续动作取消，下一步从实测关节重新规划。

Stack 的 `mark_span` 只对 VISTA 选择的四个毫米边界取中点。

## 在 GPU 上训练 ACT

从仓库根目录运行。`--out run01 --task lift` 读取 `data/run01/manifest.json` 和 `data/run01/lift/`，权重写入 `output/act/run01/lift/`。换一个任务名，就会在同一实验下独立训练该任务的权重。

```bash
python -m pip install -r requirements-act.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m act.train --out run01 --task lift
python -m act.train --out run01 --task stack
python -m act.train --out run01 --task sort
```

训练需要 CUDA 版 PyTorch 和匹配的 torchvision，第二条命令应输出 `True`。初次训练使用 torchvision 的 ImageNet ResNet-18 权重，空缓存的机器首次启动时需要能下载权重。租用 GPU 时携带项目代码及完整的 `data/run01/` 即可。

| 参数 | 默认值与用途 |
| --- | --- |
| `--data` | `data/<out>`。若想用同一批示范另开实验，可用 `--out experiment2 --data data/run01`。 |
| `--epochs` / `--batch-size` | 100 / 8。续训时 epochs 表示总轮数，batch size 沿用初次训练的值。 |
| `--workers` | 2。每个数据加载器用于读取 HDF5 和缩放相机图像的进程数，0 表示在训练进程中读取。 |
| `--resume` | 从同名任务实验的 `last.pt` 续训。实验目录已存在时，新训练需要另取名字。 |

```bash
python -m act.train --out run01 --task lift --resume --epochs 150
```

模型使用共享的 ResNet-18 逐相机编码图像，将实测关节状态和 CVAE 潜变量送入 Transformer，一次预测 32 拍关节目标。Lift 和 Stack 用两路相机、6 维目标；Sort 用三路相机、按左臂再右臂排列的 12 维目标。每局的回位拍也参与监督。

每次新训练从已成功导出的示范中按**整局**留约 20% 验证。按 seed 排序后分段固定抽样，让验证集覆盖较早和较晚采集的局。训练集单独计算关节状态与动作的归一化统计。动作片段到本局末尾即填充，填充位置不计入 L1。每轮在留出集上计算 `L1 + 10 × KL`，固定验证的潜变量抽样，按最低验证损失选 `best.pt`，平分取较早轮次。真实执行时潜变量取零。任务成功率须另做闭环评估。[ACT 原论文与示意图](https://tonyzhaozh.github.io/aloha/) · [作者的损失实现](https://github.com/tonyzhaozh/act/blob/main/policy.py)

`output/act/run01/<task>/` 中的 `training_data.json` 保存训练与验证 seed、相对路径、哈希和归一化统计，`epoch_*.json` 保存两组的 L1、KL 与总损失，`best.pt` 用于推理，`last.pt` 同时保存优化器和随机状态。续训读取初次保存的清单并核对哈希；想纳入后来补录的示范时，使用新的实验名。

## 评估 best.pt

在 MuJoCo 里从随机初姿闭环执行 ACT，并为每个测试 seed 保存同步相机录像。准备好 `output/act/run01/lift/best.pt`，从项目根目录运行。独立的评估机器只需要项目代码、依赖和该权重。无窗口 Linux 可先设置 `MUJOCO_GL=egl`。

```bash
python -m pip install -r requirements-act.txt -r requirements.txt
python -m evaluation.rollout --out run01 --task lift
```

把 `lift` 换成 `stack` 或 `sort` 即可评估对应的权重。默认从 seed 1000 到 1099（含两端），共 100 局、100 个 MP4，最多 10 个仿真进程共享一个批量推理模型。有 CUDA 时使用 GPU 推理。小规模试跑可用 `--seed 1000 --episodes 5 --threads 2`。测试 seed 与权重记录的训练或验证 seed 重合时会报错。

每局从第 0 帧的随机臂姿开始，由 ACT 自己完成回位。从实测关节与原始相机图像预测动作，执行训练协议指定的前 8 拍，再观察并重新预测。每个 10 Hz 控制拍调用该场景自己的裁判，成功立即结束。Lift 最多 120 拍，Stack 和 Sort 最多 300 拍。当前 Sort 的成功示范中，seed 10 和 14 分别用了 336 和 408 拍，超过这项评估的 30 秒时限。

每次评估都在 `output/eval/<out>/<task>/seeds-<首>-<尾>-<唯一编号>/` 新建目录。每局保存 `seed-1000-success.mp4` 或 `seed-1000-failure.mp4` 和对应的 `seed-1000.json`，后者含实际关节目标与逐拍裁判结果。录像从初始帧录到终态，帧数等于控制拍数加一。全部 seed 完成后才写 `summary.json`，给出成功数、总局数、成功率、逐局结果、录像文件名和权重 SHA-256。中断时已完成的局保留在本次评估目录。

## 仓库文件

```text
.env.example            .gitignore            README.md
requirements.txt        contracts.py          run_collect.py
control/                SO-101 接口、MuJoCo 执行器、动作、夹持与稳定性判定
scene/                  桌面与相机、Lift / Stack / Sort、GUI、自检与回放
  tasks.py              采集与闭环评估共用的任务注册表
  assets/so101/         XML、两张方块贴图、18 个模型网格、来源与许可
brains/vista/           决策图、模型适配、工具、经验与文件锁
  prompts/              lift.txt、stack.txt、sort.txt
act/                    数据、模型、离线训练与验证、并行推理与仿真 worker
evaluation/rollout.py   从 best.pt 运行闭环测试并逐局录像
requirements-act.txt    GPU 训练依赖
collection/
  batch.py              具名批次、seed、尝试记录与调度规则
  recorder.py           逐拍记录和 HDF5 发布
tests/
  test_collection_batch.py
  test_tabletop_tasks.py
  test_act_contract.py
  test_act_training.py
  test_act_rollout.py
```

运行 `python -m unittest discover -s tests -p "test_collection_batch.py"` 可以在临时目录、使用离线模型验证续录、扩容、补录和导出。运行 `python -m unittest discover -s tests -p "test_tabletop_tasks.py"` 可验证场景和控制边界。安装两份依赖后，`python -m unittest discover -s tests -v` 可运行完整测试，含真实 MuJoCo 环境的多进程评估与逐帧视频校验。实际数据、权重、在线日志及 API 密钥均保存在本地。
