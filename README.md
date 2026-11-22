# SO-101 · MuJoCo · VISTA 示范采集

右臂抬红块（Lift）、右臂把红块叠到绿块上（Stack）、双臂分拣（Sort）。三个任务共用一张桌子、两台 SO-101 和一套 10 Hz 控制接口。VISTA 根据相机画面和手臂状态操作，采集器保存每拍的观测与关节目标，并把成功示范导出为 HDF5。

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

在仓库根目录把 `.env.example` 复制成 `.env` 并填写密钥。使用 DeepSeek 时填写 `DEEPSEEK_API_KEY`。使用官方 OpenAI SOL 时填写 `OPENAI_BASE_URL=https://api.openai.com/v1`、`OPENAI_MODEL=gpt-6.2-sol`、`OPENAI_REASONING_EFFORT=xhigh` 和 `OPENAI_API_KEY`。SOL 通过 Responses API 接收多相机图像和工具回执。

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

续录继承批次的任务、模型设置、任务提示和 play 上限。若一次扩容执行到一半退出，用 `--resume NAME --fix` 接续；再次传 `--add N` 表示另外新增 N 个 seed。达到 play 上限属于一次已完成的策略试验，成功或正常失败的 seed 都跳过。每个 seed 的首次实际录制叫 `attempt_01`，手动 `--fix` 重新启动该 seed 才会产生下一个编号。单局内重试模型请求沿用当前 attempt。局末复盘错误保留在已完成的记录中。若物理任务已经成功而 HDF5 导出失败，`--fix` 从该次 raw 重建文件。遇到已有文件与记录不一致时会报错并保留现场。

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

批次 `run01` 使用 SOL xhigh，每局上限 24 轮 play，三个任务各跑 seed 0–9。

| 任务 | 首试成功 | 补录 | 已保存示范 |
| --- | ---: | ---: | ---: |
| Lift | 10/10 | — | 10 |
| Stack | 10/10 | — | 10 |
| Sort | 3/10 | 7/7 | 10 |

Sort 首轮有 7 局在动作期间收到 HTTP 429，原记录保存在对应 seed 的 `attempt_01`。双并发补录成功的 7 局保存在 `attempt_02`。该任务一共启动 17 次，10 个 seed 最终都有示范。

## 控制与模型工具

`move_by(dx_mm,dy_mm,dz_mm)` 从当前实测 TCP 起步，每轴至多 100 mm；可选 `grasp_yaw_deg` 是相对基座中性抓取朝向的绝对角，范围 ±45°，省略时保持上一朝向。控制层负责逆运动学和逐拍插值。双臂操作要指定活动臂，另一臂保持当前关节目标。闭爪只返回夹稳与否，抬升后由 VISTA 看图复核。未稳定会收到 `SETTLE_TIMEOUT`，下一次用 `observe` 或续发同一夹爪指令等待。

首轮另附一张带毫米刻度的固定相机副本，供模型目视读取桌上方块的顶面位置。标尺根据相机标定和固定基座绘制。示范保存同步相机的原始 RGB。`read_ruler` 可按模型指定的毫米范围逐点读取同一张 front 原图的 RGB，由模型自己判断颜色、方位和步幅。若下降受阻且实测关节和图像已稳定，允许原朝向纯竖直上退重新对准。已静止而实测仍未到指定位置的移动报 `MOTION_BLOCKED`，整批后续动作取消，下一步从实测关节重新规划。

Stack 的 `mark_span` 只对 VISTA 选择的四个毫米边界取中点。

## 仓库文件

```text
.env.example            .gitignore            README.md
requirements.txt        contracts.py          run_collect.py
control/                SO-101 接口、MuJoCo 执行器、动作、夹持与稳定性判定
scene/                  桌面与相机、Lift / Stack / Sort、GUI、自检与回放
  assets/so101/         XML、两张方块贴图、18 个模型网格、来源与许可
brains/vista/           决策图、模型适配、工具、经验与文件锁
  prompts/              lift.txt、stack.txt、sort.txt
collection/
  batch.py              具名批次、seed、尝试记录与调度规则
  recorder.py           逐拍记录和 HDF5 发布
tests/
  test_collection_batch.py
  test_tabletop_tasks.py
```

运行 `python -m unittest discover -s tests -p "test_collection_batch.py"` 可以在临时目录、使用离线模型验证续录、扩容、补录和导出。运行 `python -m unittest discover -s tests -p "test_tabletop_tasks.py"` 可验证场景和控制边界。实际数据、在线日志、API 密钥及本地训练代码都由 Git 忽略。
