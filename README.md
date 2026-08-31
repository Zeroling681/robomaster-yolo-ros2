# 桌面物体目标检测实验

开发环境运行在 WSL 2 的 Ubuntu 22.04 中，Ubuntu 虚拟磁盘位于
`F:\WSL\Ubuntu-22.04`。已配置 ROS 2 Humble、Cyclone DDS、视觉消息、
OpenCV、PyTorch CUDA 和 Ultralytics。

当前验证版本：Python 3.10.12、ROS 2 Humble、PyTorch 2.12.1+cu126、
Ultralytics 8.4.127、OpenCV 4.10.0、NumPy 1.26.4。

## 进入并检查环境

在 PowerShell 中进入 Ubuntu：

```powershell
wsl -d Ubuntu-22.04
```

然后在 Ubuntu 中运行：

```bash
cd /mnt/f/PycharmProjects/robomaster
source scripts/activate_wsl.sh
python scripts/check_env.py
python scripts/check_yolo.py
bash scripts/check_ros2.sh
```

三个检查分别输出 `ENVIRONMENT_CHECK=PASS`、`YOLO_GPU_INFERENCE=PASS`
和 `ROS2_PUB_SUB=PASS`，表示 Python/GPU、YOLO 实际推理及 ROS 2 话题
互通均可用。

## 关键路径

- Python 虚拟环境：`/home/tonyt/.venvs/robomaster`
- ROS 2：`/opt/ros/humble`
- Windows 项目目录：`F:\PycharmProjects\robomaster`
- Ubuntu 中的项目目录：`/mnt/f/PycharmProjects/robomaster`
- 视频抽帧与标注说明：`dataset_work/README.md`
- X-AnyLabeling 操作规范：`dataset_work/LABELING_GUIDE.md`

## 困难样本人工复核

2026-08-30 在 X-AnyLabeling 中复核了以下样本：

- `hard_robot_mouse.jpg`：复核确认画面中没有鼠标，按背景样本处理，保持
  JSON 的 `shapes` 为空，不补绘目标框。该图片保留为鼠标难负样本。
- `neg_laptop.jpg`：左侧边缘可见一个杯子，仅标注这一个 `cup` 框；笔记本、
  键盘和屏幕内容不标注。该图片同时作为杯子正样本和鼠标难负样本。

修改标注后，应重新运行 YOLO 转换和验证脚本，再开始下一轮训练；不要直接
手工修改导出的坐标文件。

## 困难负样本补充（v7）

2026-08-30 补充 20 张实验室场景的空标注负样本，覆盖机器人、黑色座椅、
手机、窗户、桌面、线缆和远景场景，用于降低“黑色物体被误识别为鼠标”的概率。
这些样本仅进入训练集；原始图片不上传 GitHub。发现画面中确有鼠标的
`c388d96805d52dd6e8cee9daea07f2ed.jpg` 已排除，避免将真实鼠标错误地作为背景。

## 外接摄像头适配数据集（v9）

为解决旧模型在外接摄像头画面中对鼠标和保温杯漏检的问题，2026-08-31
将一段独立的外接摄像头录制视频人工复核后并入数据集。审计集位于
`dataset_work/audit_dataset_v9/`，YOLO 导出集位于
`dataset_work/audit_dataset_v9/yolo_export/`。

- 审计集：1,098 张图片；其中 36 张为外接摄像头新样本。
- 新相机样本：33 个 `mouse` 框、21 个 `cup` 框、2 张空标注负样本。
- 可训练导出集：637 张图片，按 `train/val/test = 336/61/240` 划分；
  共 800 个目标框（鼠标 495、杯子 305），另有 32 张负样本。
- 审计集中保留 461 张 `excluded` 样本作为可追溯记录；导出器不会把它们写入
  训练、验证或测试目录。

合并和导出均由下列命令生成，运行前需先在 X-AnyLabeling 中复核相机批次：

```powershell
py -3.13 scripts/merge_camera_annotation_batch.py `
  --audit dataset_work/audit_dataset_v8 `
  --batch dataset_work/camera_v9_annotation_batch `
  --output dataset_work/audit_dataset_v9
py -3.13 scripts/export_audited_yolo.py `
  --audit dataset_work/audit_dataset_v9 `
  --output dataset_work/audit_dataset_v9/yolo_export
py -3.13 scripts/verify_annotation_report.py `
  dataset_work/audit_dataset_v9/yolo_export/export_report.json
```

## 数据清洗与初始标注工具

为方便复现 v8 到 v10 的数据清洗过程，项目保留了两项独立工具。脚本只生成
新的审计结果，不会直接删除原始图片。

`exclude_oversized_training_samples.py` 用于排除训练集中占画面比例过大的目标框。
默认以 v7 审计集为输入，将目标框面积超过图片面积 65% 的训练样本标记为
`excluded_oversized_target`，并生成 v8 审计集和 `cleaning_report.json`：

```powershell
py -3.13 scripts/exclude_oversized_training_samples.py `
  --audit dataset_work/audit_dataset_v7 `
  --output dataset_work/audit_dataset_v8 `
  --max-train-box-area 0.65
```

`apply_camera_v10_initial_labels.py` 为外接摄像头 v10 批次写入保守的第一轮人工框，
重点覆盖蓝色保温杯和旧模型漏检的鼠标角度：

```powershell
py -3.13 scripts/apply_camera_v10_initial_labels.py `
  --batch dataset_work/camera_v10_annotation_batch
```

脚本不会把未确认帧自动当作负样本。运行后仍需使用 X-AnyLabeling 逐张复核，
确认框的位置、类别和遗漏目标，再执行合并与 YOLO 导出。

## v8 置信度评估

v8 模型保留了三组独立测试命令，用同一测试集比较不同置信度阈值对误检和
漏检的影响：

```bash
bash scripts/evaluate_v8_conf025_wsl.sh  # conf=0.25，偏向召回率
bash scripts/evaluate_v8_conf035_wsl.sh  # conf=0.35，折中设置
bash scripts/evaluate_v8_wsl.sh          # conf=0.60，偏向精确率
```

三个脚本均使用 `imgsz=768`、`iou=0.45` 和 GPU 0，结果写入不同的运行目录，
避免互相覆盖。评估时应同时查看每类 TP、FP、FN 和错误样本图片，不能只比较
单个总分。

## YOLO 版本与任务配置

本项目使用 Ultralytics `8.4.127` 提供的 YOLO11 检测接口，基础网络为
YOLO11n，初始权重文件是 `/home/tonyt/models/yolo11n.pt`。任务为两类目标检测：

```yaml
names:
  0: mouse
  1: cup
```

训练入口为 `scripts/train_yolo.py`。它负责检查数据配置和初始权重、加载
`ultralytics.YOLO`，统一设置 GPU、随机种子、缓存和输出目录；各数据版本的
Shell 脚本负责传入本轮实验参数。环境中的主要固定版本为：

- Python `3.10.12`
- PyTorch `2.12.1+cu126`
- Ultralytics `8.4.127`
- OpenCV `4.10.0`
- NumPy `1.26.4`

可在 WSL 虚拟环境中安装或恢复 YOLO 依赖：

```bash
cd /mnt/f/PycharmProjects/robomaster
source /home/tonyt/.venvs/robomaster/bin/activate
python -m pip install -r requirements-yolo.txt
python scripts/check_yolo.py
```

## v7 至 v10 训练参数

各版本使用的完整入口参数如下。`lr0=auto` 表示脚本不显式传入初始学习率，
由 Ultralytics 的 `optimizer=auto` 决定。v7、v8 的 `args.yaml` 虽然记录
`lr0=0.01`，但自动优化器可以在启动时重新选择优化器和有效学习率，应以训练
日志中的 optimizer 输出为准。

| 版本 | 数据配置 | 初始权重 | epochs | patience | batch | imgsz | optimizer | lr0 | lrf | 输出名称 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| v7 | `dataset_work/yolo_export_v7/dataset.yaml` | `yolo11n.pt` | 100 | 25 | 16 | 768 | `auto` | auto | 0.01 | `mouse_cup_yolo11n_v7_768` |
| v8 | `dataset_work/yolo_export_v8/dataset.yaml` | `yolo11n.pt` | 100 | 25 | 16 | 768 | `auto` | auto | 0.01 | `mouse_cup_yolo11n_v8_768` |
| v9 | `dataset_work/audit_dataset_v9/yolo_export/dataset.yaml` | v8 `best.pt` | 50 | 15 | 16 | 768 | `SGD` | 0.001 | 0.01 | `mouse_cup_yolo11n_v9_camera_768` |
| v10 | `dataset_work/audit_dataset_v10/yolo_export/dataset.yaml` | v9 `best.pt` | 60 | 20 | 16 | 768 | `SGD` | 0.0005 | 0.01 | `mouse_cup_yolo11n_v10_clean_768` |

`train_yolo.py` 对所有版本统一传入以下参数：

| 参数 | 数值 | 作用 |
| --- | --- | --- |
| `device` | `0` | 使用第一块 CUDA GPU |
| `workers` | `4` | 数据加载进程数 |
| `cache` | `ram` | 将训练图片缓存到内存 |
| `pretrained` | `True` | 从指定权重继续训练 |
| `seed` | `20260824` | 固定随机种子 |
| `deterministic` | `True` | 尽量保证实验可复现 |
| `close_mosaic` | `10` | 最后 10 轮关闭 Mosaic |
| `amp` | `True` | 启用混合精度训练 |
| `plots` | `True` | 输出曲线、混淆矩阵等图片 |
| `project` | `runs/detect` | 训练结果根目录 |
| `exist_ok` | `False` | 防止覆盖同名实验目录 |

实际训练产生的 `args.yaml` 还记录了以下 Ultralytics 参数。当前 v10 使用
`momentum=0.937`、`weight_decay=0.0005`、`warmup_epochs=3.0`、
`warmup_momentum=0.8`、`warmup_bias_lr=0.1`；检测损失权重为 `box=7.5`、
`cls=0.5`、`dfl=1.5`。

默认训练增强为 `hsv_h=0.015`、`hsv_s=0.7`、`hsv_v=0.4`、
`translate=0.1`、`scale=0.5`、`fliplr=0.5`、`mosaic=1.0` 和
`erasing=0.4`。当前没有启用额外旋转、剪切、透视、上下翻转、MixUp、
CutMix 或 Copy-Paste，即 `degrees=0`、`shear=0`、`perspective=0`、
`flipud=0`、`mixup=0`、`cutmix=0`、`copy_paste=0`。

## 训练启动方式

当前推荐训练版本是 v10。在 PowerShell 中进入 WSL 后启动：

```powershell
wsl -d Ubuntu-22.04
```

```bash
cd /mnt/f/PycharmProjects/robomaster
source /home/tonyt/.venvs/robomaster/bin/activate
bash scripts/train_v10_camera_wsl.sh
```

也可以直接从 PowerShell 用一条命令启动：

```powershell
wsl -d Ubuntu-22.04 -- bash -lc `
  "cd /mnt/f/PycharmProjects/robomaster && bash scripts/train_v10_camera_wsl.sh"
```

需要复现历史实验时，分别运行：

```bash
bash scripts/train_v7_wsl.sh
bash scripts/train_v8_wsl.sh
bash scripts/train_v9_camera_wsl.sh
bash scripts/train_v10_camera_wsl.sh
```

Shell 脚本会先进入对应 YOLO 导出目录，因为 `dataset.yaml` 使用 `path: .`，
必须从数据集目录解析 `images/train`、`images/val` 和 `images/test`。训练结果保存
在 `runs/detect/<输出名称>/`，主要文件包括：

- `weights/best.pt`：验证指标最佳的 PyTorch 权重；
- `weights/last.pt`：最后一轮权重；
- `args.yaml`：本次训练最终采用的全部参数；
- `results.csv` 和 `results.png`：逐轮指标及训练曲线；
- `confusion_matrix.png`：验证集混淆矩阵。

这些脚本中的虚拟环境和 `/mnt/f/PycharmProjects/robomaster` 路径对应当前 WSL
部署；复制到其他电脑时，需要先修改为实际项目路径和虚拟环境位置。v9、v10
是微调流程，启动前还必须确认上一版本的 `weights/best.pt` 已存在。

## sudo 密码

用户 `tonyt` 当前不启用免密 sudo。需要自行设置密码时，在 PowerShell 中运行：

```powershell
wsl -d Ubuntu-22.04 -u root passwd tonyt
```
