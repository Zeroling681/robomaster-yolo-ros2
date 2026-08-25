# 数据标注与质量控制

## 类别定义

本实验使用两类桌面物体：

| 类别 ID | 类别名 | 标注原则 |
| --- | --- | --- |
| 0 | `mouse` | 标注完整鼠标外轮廓，不包含鼠标线缆 |
| 1 | `cup` | 标注杯体可见区域，遮挡时只标注可见部分 |

标注工具为 X-AnyLabeling，导出格式为 LabelMe/X-AnyLabeling JSON，之后由
`scripts/prepare_yolo_dataset.py` 转换为 YOLO 检测格式。

## 人工复核流程

1. 先对视频抽帧结果进行人工筛选，删除模糊、严重遮挡和重复帧。
2. 使用矩形框标注每个目标，类别名称只允许使用 `mouse` 和 `cup`。
3. 完成一轮标注后，用 `scripts/visualize_yolo_labels.py` 生成复核图。
4. 检查越界框、空标签、类别拼写和同一目标重复框。
5. 由 `scripts/prepare_yolo_dataset.py` 生成训练、验证和测试清单，并保留
   `split_manifest.csv` 作为划分记录。

## 数据划分

按照视频来源划分数据，而不是把同一视频的相邻帧随机分到不同集合，避免
测试集与训练集出现近似帧。目标比例为训练集 70%、验证集 20%、测试集 10%。

## 需要特别核对的事项

Jetson 上已有的 `/home/nvidia/jetson_yolo/best.pt` 加载后显示类别为
`mouse` 和 `bottle`。这与本项目数据集的 `mouse` 和 `cup` 命名不一致；在
最终验收前必须确认该权重是否为本项目最新权重，不能仅靠类别下标推断两者等价。
