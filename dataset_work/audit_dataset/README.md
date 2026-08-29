# 数据集审计目录

这个目录用于在 X-AnyLabeling 中统一检查当前训练数据。类别固定为：

- `mouse`
- `cup`

目录内容：

- `images/`：所有待审计图片，文件名全局唯一；
- `annotations/`：与图片同名的 X-AnyLabeling JSON 标注；
- `source_videos/`：本轮新增的困难样本原始视频；
- `audit_manifest.csv`：每张图片的来源、划分、标注来源和审核状态；
- `audit_report.json`：数量与标签合法性汇总；
- `classes.txt`：类别顺序。

启动方法：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/open_audit_dataset.ps1
```

审核规则：

1. 所有真实出现的鼠标和杯子都必须标框；
2. 黑色干扰物不是鼠标，不要给干扰物标框；
3. 遮挡时只框可见目标，边界尽量贴紧；
4. 模糊但仍可确定类别的目标需要标框；
5. 完全无法确认类别的帧可删除，并在清单中记录；
6. `annotation_required` 必须优先处理；
7. `review_required` 是自动或继承标注，必须人工确认；
8. 审核完成后在 X-AnyLabeling 中将图片标记为已检查。

注意：当前目录是审计工作集，不应在审核完成前直接用于重新训练。
