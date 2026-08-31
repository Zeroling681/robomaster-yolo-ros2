# v11 错误反馈审计数据集

基础审计集为 `audit_dataset_v10`，新增人工复核批次为
`camera_v11_error_corrections`。新增样本只进入训练集，验证集和测试集沿用原有
独立场景，避免同一段摄像头录像跨集合泄漏。

本轮新增两张训练图片：

- `v10_error_missed_cup_mouse_t090000.jpg`：为运动模糊下漏检的杯子和同画面鼠标
  分别补框；
- `v10_error_false_mouse_background_t092000.jpg`：从误检画面裁出不含真实目标的
  左侧干扰区域，使用空标签作为负样本。

导出后的数据规模为 train 392、val 61、test 240，共 693 张图片；其中空标签
负样本 36 张，鼠标框 546 个，杯子框 349 个。`audit_yolo_export.py` 检查通过，
未发现跨集合重复、场景泄漏、越界框或非法类别。
