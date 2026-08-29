# v5 YOLO 导出说明

该目录保存本次人工复核后的 YOLO 标签、划分清单和导出报告。

- `export_manifest.csv` 记录每张图片的划分和审核状态。
- `labels/` 保存 `train/val/test` 标签。
- 原始图片统一保存在上级 `../images/`，没有再次复制到本目录，避免 Git LFS 重复存储。
- 需要生成可直接训练的目录时，按清单将 `../images/` 中的同名图片复制到 `images/train`、`images/val`、`images/test`，或使用项目中的数据整理脚本。

`dataset.yaml` 使用相对路径，便于在 Windows、WSL 和 Jetson 间迁移。
