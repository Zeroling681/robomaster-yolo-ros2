# 实验过程记录

本文档只记录已经实际执行过的步骤；后续实验请继续追加日期、命令和结果。

## 2026-08-25：Jetson 环境与实时推理验证

设备信息：

- Jetson Orin NX 16GB
- JetPack 6.2.1，CUDA 12.6
- Ubuntu 22.04，Python 3.10.12
- PyTorch 2.5.0a0（CUDA 12.6）
- Ultralytics 8.4.128
- OpenCV 4.13.0

启动环境：

```bash
source /home/nvidia/activate_yolo.sh
```

模型：

```text
/home/nvidia/jetson_yolo/best.pt
```

模型加载结果：

```text
classes: {0: mouse, 1: bottle}
torch.cuda.is_available(): True
```

摄像头 `/dev/video0` 能读取 640x480 图像。使用 CUDA 进行实时推理，处理约
1390 帧后手动停止，终端记录的累计速度约为 23.8 FPS。结果视频保存为：

```text
/home/nvidia/jetson_yolo/runs/camera_test/0.avi
```

该视频已复制到本地 `results/jetson_camera_test.avi`。本次记录用于验证环境和
实时链路，不等同于最终的 20 个物体准确率测试。

## 2026-08-29：v5-2 困难样本训练与留出测试

训练数据来自 `dataset_work/audit_dataset/yolo_export_v5`，共 1036 张图像，
其中训练集 980 张、验证集 38 张、测试集 18 张；类别为 `mouse` 和 `cup`。
训练输出保存在本机 `runs/detect/mouse_cup_yolo11n_v5-2`，权重文件没有纳入
版本库，使用时通过本地路径传入脚本。

验证集最佳记录（第 65 个 epoch）：

```text
precision: 0.95258
recall:    0.99985
mAP50:     0.98532
mAP50-95:  0.84422
```

固定测试集统计（conf=0.50，IoU=0.50）：18 张图像、26 个真实目标，预测 26
个，TP=23、FP=3、FN=3，按目标计正确率为 23/26=88.46%。其中鼠标 TP=14、
FP=3、FN=3；水杯 TP=9。错误主要集中在黑色背景物体造成的鼠标误检，以及杯子
遮挡时的鼠标漏检，已在 `runs/detect/mouse_cup_yolo11n_v5-2_final_test/errors`
保存典型图片。

## 2026-08-30：Windows 外接摄像头测试

使用电脑外接摄像头索引 1 和 v5-2 ONNX 导出模型进行连续测试，命令入口为：

```powershell
py -3.13 scripts/live_camera_onnx.py --camera 1 `
  --model results/mouse_cup_yolo11n_v5-2_best.onnx `
  --conf 0.60 --save results/live_camera_external1_v5-2_conf060.avi
```

原始录像仅保存在本机 `results/`，仓库通过 `.gitignore` 排除 AVI 和 ONNX 大文件；
本记录保留命令和文件名，便于复现实验。

## 后续记录模板

```text
日期：
模型权重：
测试样本数：
正确数：
错误数：
准确率：
平均 FPS：
典型错误：
```
