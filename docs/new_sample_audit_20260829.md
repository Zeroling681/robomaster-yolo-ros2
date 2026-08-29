# 2026-08-29 补充照片审计记录

本批收到 8 张照片，原图尺寸均为 1279×1706。按 SHA256 检查后，
`codex-clipboard-c3dd7ff7-4fee-4c39-9c1c-6ed8e7cc3dfc.jpg` 与
`codex-clipboard-7af06ba9-bfe8-493d-a7f1-5a9bb7dae3e0.jpg` 完全重复，未重复入库。

## 入库结果

| 文件 | 当前状态 | 处理说明 |
| --- | --- | --- |
| `neg_new_monitor_screen.jpg` | `human_checked_negative` | 显示器/桌面场景，确认没有鼠标或水杯，保留空 JSON |
| `hard_robot_mouse.jpg` | `annotation_required` | 画面下方可见白色鼠标，需要人工框选 |
| `hard_monitor_mouse_cup.jpg` | `annotation_required` | 桌面上方可见鼠标，显示器后方有透明杯状物，需逐一确认并框选 |
| `hard_monitor_pink_mouse.jpg` | `annotation_required` | 右侧可见粉色鼠标，需人工框选 |
| `hard_keyboard_mouse.jpg` | `annotation_required` | 右下可见黑色鼠标，需人工框选 |
| `hard_foam_mouse.jpg` | `annotation_required` | 泡沫包装场景中存在鼠标/疑似目标，需人工确认 |
| `hard_foam_object_review.jpg` | `annotation_required` | 泡沫与桌面干扰场景，需人工确认白色/橙色物体是否属于目标 |

`annotation_required` 图片不会被导出脚本加入训练集，直到在 X-AnyLabeling
中补框并保存。完成审核后执行：

```powershell
py -3.13 scripts/sync_audit_review.py `
  --subset dataset_work/audit_dataset/review_missing `
  --confirm-all-reviewed
py -3.13 scripts/build_audit_views.py --force
py -3.13 scripts/export_audited_yolo.py --force
```

然后检查 `dataset_work/audit_dataset/yolo_export/export_report.json` 中
`excluded_pending` 是否为 0，再进行训练。
