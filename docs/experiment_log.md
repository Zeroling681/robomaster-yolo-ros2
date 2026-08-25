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
