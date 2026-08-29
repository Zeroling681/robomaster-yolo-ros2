# Jetson 实测结果

## `jetson_camera_test.avi`

该视频来自 Jetson Orin NX 的 `/dev/video0` 摄像头，使用
`/home/nvidia/jetson_yolo/best.pt` 进行 CUDA 推理并保存检测框。

- 输入分辨率：640x480
- 处理帧数：约 1390 帧
- 停止前累计速度：约 23.8 FPS
- 终端中观察到的类别：`mouse`、`bottle`

本文件用于记录环境和实时链路验证。最终验收的 20 个物体准确率应另行使用
固定测试清单统计，不能用本段视频的 FPS 结果代替准确率结论。

## v5-2 Windows 外接摄像头

2026-08-30 使用 `camera 1` 运行 v5-2 ONNX 模型，输出文件为
`live_camera_external1_v5-2_conf060.avi`。该录像没有提交到 GitHub，保留在本机
用于回看误检和漏检；对应命令和测试说明见 `docs/experiment_log.md`。
