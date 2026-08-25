# 标注调试记录

本文档只记录仓库中验证报告能够追溯的现象和处理规则。修复后重新运行转换
脚本，并把新的 `validation_report.json` 一并保存。

## 1. 重复框

现象：部分 X-AnyLabeling JSON 中同一类别出现两个高度重叠的矩形框，通常来自
辅助框和人工框同时保留。重复框会使一个目标被计数两次，并干扰训练指标。

处理：`prepare_yolo_dataset.py` 对同类别框计算 IoU；当 IoU 达到 `0.9` 时只
保留先出现的框，并把删除的框写入 `removed_duplicate_boxes`。

当前验证报告记录了 6 个被删除的重复框，IoU 范围约为 0.943～0.986。

## 2. 越界框

现象：靠近画面边缘的人工框可能超出图片宽高范围。

处理：转换前把坐标裁剪到 `[0, width]` 和 `[0, height]`；裁剪后面积为零的框
直接报错，不生成无效 YOLO 标签。被裁剪的框写入 `clipped_boxes`，便于回到
原始 JSON 复查。

当前验证报告记录了 8 个越界框，未发现跨类别重叠框。

## 3. 漏标和错误类别

现象：辅助标注可能完全漏掉主目标，或把鼠标和杯子标成相反类别。当前清单中
有 40 张图片被列为优先人工复核对象；这些图片不能直接作为训练真值。

处理：

- X-AnyLabeling 复核时逐张检查所有可见目标；
- 转换脚本要求每张图片至少包含其主场景类别，缺失时直接失败并指出文件名；
- `verify_annotation_report.py` 在训练前检查总数、数据划分和跨类别重叠；
- 复核后重新生成 YOLO 标签，不手工修改导出的坐标文件。

## 当前复核结果

```text
image_count: 188
raw_box_count: 239
exported_box_count: 233
train/val/test: 132/38/18
cross_class_overlaps: 0
```

运行质量检查：

```bash
python scripts/verify_annotation_report.py \
  dataset_work/yolo_dataset_v2/validation_report.json
```
