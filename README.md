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

## v7 至 v9 训练复现

历史训练参数分别保存在独立 WSL 脚本中：

```bash
bash scripts/train_v7_wsl.sh
bash scripts/train_v8_wsl.sh
bash scripts/train_v9_camera_wsl.sh
```

- v7：从 YOLO11n 预训练权重开始，以 `imgsz=768` 训练 100 轮。
- v8：使用清理后的 v8 导出集，以相同基础参数重新训练 100 轮。
- v9：从 v8 最佳权重继续微调外接摄像头数据，训练 50 轮，并使用较低学习率
  `lr0=0.001`，减少已学习特征被快速破坏的风险。

这些脚本中的虚拟环境和 `/mnt/f/PycharmProjects/robomaster` 路径对应当前 WSL
部署；复制到其他电脑时，需要先修改为实际项目路径和虚拟环境位置。

## sudo 密码

用户 `tonyt` 当前不启用免密 sudo。需要自行设置密码时，在 PowerShell 中运行：

```powershell
wsl -d Ubuntu-22.04 -u root passwd tonyt
```
