# Mouse–Cup Object Detection (YOLO11n + ROS 2)

本项目完成了一个面向桌面场景的双类别目标检测实验。检测类别为 `mouse` 和
`cup`，包含数据采集与人工审核、YOLO 格式导出、YOLO11n 训练、Windows
ONNX 实时推理、Jetson CUDA 推理以及 ROS 2 结果发布。

## 1. 实验结果概览

| 项目 | 结果 | 说明 |
| --- | ---: | --- |
| 检测类别 | 2 | `mouse`、`cup` |
| 二十角度严格场景准确率 | 18/20 = 90% | 见 `results/v13_submission_review/v13_20_angle_results_with_horizontal_cup.csv` |
| Jetson 实时速度 | 15.7–21.2 FPS | 证据帧平均约 17.9 FPS，高于 5 FPS 要求 |
| ROS 2 发布频率 | 约 20.57 Hz | 话题 `/yolo/detections` |
| 平放杯子 | 已覆盖 | v13 数据集加入平放姿态，证据见 release 视频包 |

二十角度测试采用严格场景判定：画面中应出现的目标必须全部检测到，且不得产生
额外误检框。两项漏检被保留为错误案例，便于后续数据清洗和复训。上述准确率与
Jetson 速度来自不同证据记录，提交时应同时附上对应的视频或运行日志。

## 2. 环境要求

### 2.1 Windows 本地推理环境

- Windows 10/11
- Python 3.13（命令行使用 `py -3.13`）
- OpenCV、NumPy、ONNX Runtime
- 摄像头索引按系统实际情况设置，本文示例使用 `1`

安装基础依赖：

```powershell
py -3.13 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install opencv-python numpy onnxruntime
```

### 2.2 WSL 2 训练环境

- WSL 2 + Ubuntu 22.04
- NVIDIA GPU、CUDA 可用的 PyTorch
- Ultralytics `8.4.127`
- Python 3.10/3.11 均可用于训练脚本

安装 Python 依赖：

```bash
python3 -m venv ~/.venvs/robomaster
source ~/.venvs/robomaster/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-yolo.txt
```

检查训练环境：

```bash
python scripts/check_yolo.py
```

`scripts/check_env.py` 还会检查 CUDA、ROS 2、`cv_bridge` 和
`vision_msgs`，因此应在 Jetson/ROS 2 环境中运行，而不是普通 Windows 虚拟环境。

WSL 中的 `/mnt/f/PycharmProjects/robomaster` 只是本机当前目录的一个示例，
换电脑时请将脚本中的项目路径改成实际挂载路径。

### 2.3 Jetson + ROS 2 环境

- NVIDIA Jetson（实验记录：Jetson Orin NX 16 GB）
- Ubuntu 22.04、JetPack/CUDA、CUDA 版 PyTorch
- ROS 2 Humble
- `rclpy`、`vision_msgs`、OpenCV、Ultralytics

Jetson 上的项目目录和模型路径可以自行调整。下面命令假定项目位于
`/home/nvidia/jetson_yolo`，ROS 2 工作空间位于 `~/yolo_ros2_ws`。

## 3. 项目目录

```text
robomaster/
├─ dataset_work/
│  └─ audit_dataset_v13/
│     ├─ annotations/              # 审核后的原始标注记录
│     └─ yolo_export/              # v13 YOLO 数据集（dataset.yaml）
├─ models/v13/
│  ├─ README.md                    # 权重与指标说明
│  ├─ args.yaml                    # Ultralytics 实际训练参数
│  ├─ results.csv                  # 每轮训练/验证指标
│  ├─ results.png                  # 训练曲线
│  └─ confusion_matrix.png         # 混淆矩阵
├─ scripts/
│  ├─ train_yolo.py                # 通用训练入口
│  ├─ train_v13_horizontal_cup_wsl.sh
│  ├─ live_camera_onnx.py          # Windows ONNX 摄像头推理
│  ├─ live_camera_pt.py            # Jetson PyTorch 摄像头推理
│  ├─ run_video_onnx.py            # 录制视频离线推理
│  ├─ run_jetson_camera.sh         # Jetson 摄像头启动封装
│  └─ capture_jetson_ros2_evidence.sh
├─ ros2/yolo_detection_ros2/       # ROS 2 检测节点和 launch 文件
├─ docs/                           # 实验报告、检查清单和数据说明
├─ results/                        # 本地视频、日志和评测证据
├─ release/                        # 可直接提交/下载的压缩包与快速指南
├─ requirements-yolo.txt
└─ README.md
```

`runs/`、大型视频和部分原始图片属于生成文件，按 `.gitignore` 规则不全部纳入
版本控制。需要完整模型和视频时，请使用 `release/` 下的压缩包。

## 4. 数据集

### 4.1 类别与标注格式

`dataset_work/audit_dataset_v13/yolo_export/dataset.yaml` 定义了两个类别：

```yaml
names:
  0: mouse
  1: cup
```

每个图片对应一个同名 `.txt` 标签，行格式为
`class_id x_center y_center width height`，坐标均为相对图片宽高归一化值。
没有目标的图片保留空标签，用作负样本；误检背景经过审核后也按空标签加入。

### 4.2 数据构建流程

1. 从手机、Windows 摄像头和 Jetson 摄像头采集不同距离、光照、遮挡和姿态的画面。
2. 删除重复、模糊或无法判断的帧；保留能代表困难情况的负样本。
3. 使用预标注提高效率，再逐张人工检查类别、边界框和空标签。
4. 将审核结果导出为 YOLO 目录结构，并固定 `train/val/test` 划分。
5. v13 重点补充杯子完全平放、倾斜和不同背景，以及鼠标侧面/底部等易漏检姿态。

当前 v13 导出统计为 793 张图片：训练集 492 张、验证集 61 张、测试集 240 张。
数据集说明和审计记录见 `dataset_work/audit_dataset_v13/README.md` 及 `docs/`。

## 5. 模型说明

最终模型为 YOLO11n（轻量检测模型），输入尺寸 768×768，类别数为 2。

| 项目 | v13 配置 |
| --- | --- |
| 初始权重 | v12 `best.pt` 微调 |
| Epoch / patience | 40 / 12 |
| Batch / image size | 16 / 768 |
| 优化器 | SGD |
| `lr0` / `lrf` | 0.00015 / 0.01 |
| 动量 / 权重衰减 | 0.937 / 0.0005 |
| 最佳轮次 | epoch 27 |
| 验证集 Precision / Recall | 0.856 / 0.776 |
| 验证集 mAP50 / mAP50-95 | 0.886 / 0.715 |

本地训练完成后，权重通常位于：

```text
runs/detect/mouse_cup_yolo11n_v13_horizontal_cup_768/weights/best.pt
runs/detect/mouse_cup_yolo11n_v13_horizontal_cup_768/weights/best.onnx
```

仓库不强制跟踪大体积权重文件；可从
`release/experiment_one_v13_model.zip` 解压得到 `best.pt`、`best.onnx`、
`args.yaml` 和训练曲线。模型细节与校验和见 `models/v13/README.md`。

## 6. 安装与训练

### 6.1 直接运行 v13 训练

在 WSL 项目根目录执行：

```bash
cd /mnt/f/PycharmProjects/robomaster
source ~/.venvs/robomaster/bin/activate
bash scripts/train_v13_horizontal_cup_wsl.sh
```

该脚本使用：

```text
data:  dataset_work/audit_dataset_v13/yolo_export/dataset.yaml
model: v12 best.pt
name:  mouse_cup_yolo11n_v13_horizontal_cup_768
```

训练输出、`args.yaml`、`results.csv` 和曲线保存在 `runs/detect/` 对应实验目录。
历史 v7–v12 脚本仍保留在 `scripts/`，只用于复现实验过程。

### 6.2 手动调用通用训练入口

```bash
python scripts/train_yolo.py \
  --data dataset_work/audit_dataset_v13/yolo_export/dataset.yaml \
  --model runs/detect/mouse_cup_yolo11n_v12_phone_hard_negative_768/weights/best.pt \
  --name mouse_cup_yolo11n_v13_horizontal_cup_768 \
  --epochs 40 --patience 12 --batch 16 --imgsz 768 \
  --optimizer SGD --lr0 0.00015 --lrf 0.01
```

## 7. 启动实时检测

### 7.1 Windows：ONNX 摄像头检测与录像

先确认 `models/v13/best.onnx` 已存在（可从 release 模型包解压），然后在项目
根目录运行。`Q`、`Esc` 或关闭窗口可结束录像。

```powershell
py -3.13 scripts/live_camera_onnx.py `
  --camera 1 `
  --model models/v13/best.onnx `
  --imgsz 768 `
  --mouse-conf 0.75 `
  --cup-conf 0.75 `
  --save results/v13_camera1_detected.avi `
  --save-raw results/v13_camera1_raw.avi
```

窗口会显示类别、边界框、置信度和 FPS。`--save` 保存叠加检测结果，
`--save-raw` 保存未画框的原始视频，便于复核误检和漏检。若不需要分别设置类别
阈值，也可以使用统一的 `--conf 0.50`。

### 7.2 Windows：对已有视频进行离线推理

```powershell
py -3.13 scripts/run_video_onnx.py `
  --input results/input.avi `
  --output results/input_v13_detected.avi `
  --model models/v13/best.onnx `
  --imgsz 768 `
  --mouse-conf 0.75 `
  --cup-conf 0.75
```

### 7.3 Jetson：PyTorch/CUDA 摄像头检测

将 `best.pt` 和项目文件复制到 Jetson 后：

```bash
cd /home/nvidia/jetson_yolo
source /home/nvidia/activate_yolo.sh
python3 scripts/live_camera_pt.py \
  --camera 0 \
  --model /home/nvidia/jetson_yolo/models/v13/best.pt \
  --imgsz 768 \
  --conf 0.75 \
  --save /home/nvidia/jetson_yolo/results/v13_jetson_camera0_detected.avi
```

也可以使用启动封装脚本：

```bash
bash scripts/run_jetson_camera.sh 0
```

脚本会检查 `/dev/video0`、激活环境并优先选择 PyTorch 权重。该封装脚本的默认
权重路径仍兼容历史 v4 目录；使用 v13 时，建议优先使用上面的直接命令，或先
在脚本中将 `PT_MODEL_PATH` 和输出文件改为 v13 路径。摄像头索引为 1 时，将命令
末尾的 `0` 改为 `1`。

## 8. ROS 2 检测结果发布

ROS 2 节点每帧发布一个
`vision_msgs/msg/Detection2DArray`，其中包含类别、置信度和像素坐标框。

### 8.1 编译

```bash
mkdir -p ~/yolo_ros2_ws/src
cp -r ros2/yolo_detection_ros2 ~/yolo_ros2_ws/src/
cd ~/yolo_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 8.2 启动与验证

```bash
ros2 launch yolo_detection_ros2 detector.launch.py
```

另一终端验证话题：

```bash
source /opt/ros/humble/setup.bash
source ~/yolo_ros2_ws/install/setup.bash
ros2 topic info /yolo/detections
ros2 topic echo /yolo/detections --once
ros2 topic hz /yolo/detections
```

覆盖摄像头、模型和两个类别阈值：

```bash
ros2 run yolo_detection_ros2 detector_node --ros-args \
  -p camera:=1 \
  -p model:=/home/nvidia/jetson_yolo/models/v13/best.pt \
  -p mouse_confidence:=0.75 \
  -p cup_confidence:=0.75
```

不接摄像头时可以读取视频文件：

```bash
ros2 run yolo_detection_ros2 detector_node --ros-args \
  -p source:=/home/nvidia/jetson_yolo/results/test.avi \
  -p model:=/home/nvidia/jetson_yolo/models/v13/best.pt \
  -p show:=false
```

需要一次性生成 Jetson + ROS 2 证据时：

```bash
cd /home/nvidia/jetson_yolo
bash scripts/capture_jetson_ros2_evidence.sh 0 \
  /home/nvidia/jetson_yolo/models/v13/best.pt \
  /home/nvidia/yolo_ros2_ws
```

## 9. 结果、错误案例与提交材料

主要结果文件：

- `results/v13_submission_review/v13_20_angle_results_with_horizontal_cup.csv`：20 个视角的逐项记录及严格准确率。
- `results/v13_submission_review/v13_20_angle_evidence_with_horizontal_cup.jpg`：20 个视角证据拼图。
- `results/v13_submission_review/v13_horizontal_cup_success_detected_contact_sheet.jpg`：平放杯子成功检测案例。
- `results/` 下的 `v13_*.avi`：Windows/Jetson 检测视频和原始视频。
- `docs/Experiment_One_Object_Detection_Report.docx` 与 `.pdf`：英文实验报告。

可直接提交的打包材料：

- `release/experiment_one_v13_model.zip`：模型和训练指标。
- `release/experiment_one_v13_video_evidence.zip`：检测视频、20 角度记录和证据图。
- `release/experiment_one_v13_jetson_ros2_kit.zip`：Jetson 模型、检测程序、ROS 2 包和快速指南。
- `release/experiment_one_v13_ros2_jetson_evidence.zip`：ROS 2 话题、日志、FPS 和板端视频证据。
- `release/experiment_one_complete_submission_v13.zip`：数据集、模型、程序、结果视频、运行说明和实验报告的一体化提交包（Git LFS）。

每个压缩包的内容和 SHA-256 校验和见 `release/README.md`；Jetson 操作步骤见
`release/JETSON_ROS2_QUICKSTART.md`。

## 10. 常见问题

**为什么克隆后没有 `best.pt`？**  权重文件较大且通常被 Git 忽略。下载并解压
`release/experiment_one_v13_model.zip`，将权重放入 `models/v13/`，或在命令中直接
指定自己的权重路径。

**如何更换摄像头？**  Windows 使用 `--camera 0/1`；Jetson 使用对应的
`/dev/video0` 或 `/dev/video1`，并确认当前用户有摄像头访问权限。

**如何保存误检/漏检？**  实时检测同时使用 `--save` 和 `--save-raw`。从原始视频中
截取错误帧，人工修正漏检框；纯背景误检帧保留空标签，再按第 6 节流程重新导出和训练。

**如何只调整某一类的阈值？**  使用 `--mouse-conf` 与 `--cup-conf`；未指定的类别
回退到 `--conf`。ROS 2 节点对应参数名为 `mouse_confidence` 和 `cup_confidence`。

## 11. 相关文档

- [实验报告](docs/Experiment_One_Object_Detection_Report.pdf)
- [提交检查清单](docs/SUBMISSION_CHECKLIST.md)
- [v13 模型说明](models/v13/README.md)
- [ROS 2 节点说明](ros2/yolo_detection_ros2/README.md)
- [release 包说明](release/README.md)
