# v10 本地摄像头 20 项实物测试

## 测试配置

- 日期：2026-08-31
- 平台：Windows，本地 Python 3.13
- 摄像头：外置摄像头，索引 1
- 模型：`runs/detect/mouse_cup_yolo11n_v10_clean_768/weights/best.onnx`
- 阈值：`mouse=0.42`，`cup=0.50`
- 原始录像长度：97.4 秒，20 FPS，640×480

录像命令：

```powershell
py -3.13 scripts/live_camera_onnx.py `
  --camera 1 `
  --model runs/detect/mouse_cup_yolo11n_v10_clean_768/weights/best.onnx `
  --mouse-conf 0.42 `
  --cup-conf 0.50 `
  --save results/v10_local_camera1_20_object_take2_detected.avi `
  --save-raw results/v10_local_camera1_20_object_take2_raw.avi
```

## 统计方法

录像没有单独的按键事件标记，因此按画面切换顺序整理出 20 个有效检查点，
并在每次展示相对稳定的中点帧进行人工复核。两个多物体画面分别作为一个
检查点，其中均有 3 个杯子。第 19 项同时包含鼠标和不锈钢杯。

采用两个指标：

1. 目标实例召回：正确识别的真实目标数除以画面中的真实目标总数；
2. 严格场景正确率：一个检查点中所有真实目标均被正确识别，且没有额外误框，
   才计为正确。

## 结果

- 真实目标实例：25 个；正确识别 24 个，目标实例召回率 **96%**；
- 严格场景：20 项中 18 项完全正确，严格场景正确率 **90%**；
- 第 19 项：鼠标识别正确，但运动中的不锈钢杯漏检；
- 第 20 项：Razer 鼠标识别正确，但左侧背景出现一个额外 `mouse` 误框。

因此，本次录像满足“20 项实物检查、严格正确率不低于 80%”的阶段性要求，
但不能据此声称模型在任意场景下都达到 90%。录像中仍能看到运动模糊时的
短暂漏检和检测框闪烁，后续应继续补充运动模糊、遮挡和黑色背景负样本。

## 证据文件

- 带框录像：[`results/v10_local_camera1_20_object_test_detected.mp4`](../results/v10_local_camera1_20_object_test_detected.mp4)
- 20 项联系表：[`results/v10_local_camera1_20_object_evidence.jpg`](../results/v10_local_camera1_20_object_evidence.jpg)
- 逐项结果：[`results/v10_local_camera1_20_object_summary.csv`](../results/v10_local_camera1_20_object_summary.csv)

MP4 由原始 MJPG 录像压缩得到，保留 1948 帧、20 FPS 和 97.4 秒时长。
SHA-256：`58c4bcce4e5d2178bd5c2fe413806a805a6e1f38890dcb9dfc22bdd465c52857`。
