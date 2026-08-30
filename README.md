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

## sudo 密码

用户 `tonyt` 当前不启用免密 sudo。需要自行设置密码时，在 PowerShell 中运行：

```powershell
wsl -d Ubuntu-22.04 -u root passwd tonyt
```
