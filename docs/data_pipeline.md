# 视频抽帧与数据筛选

## 数据来源

本次数据来自 6 个视频：3 个鼠标视频和 3 个杯子视频。仓库中的
`dataset_work/video_manifest.json` 只保留视频 ID、类别和原始文件名，不提交
聊天软件的临时绝对路径。

## 筛选结果

- 正式图片：188 张
- `mouse`：93 张
- `cup`：95 张
- 原始标注框：239 个
- 去重和越界修正后导出：233 个
- 精确重复图片：0 张
- 拒绝使用的图片：4 张极小目标图，以及 1 批无效随机跳转结果

## 可复现命令

抽帧和清晰度筛选由以下脚本完成：

```bash
python scripts/select_video_frames.py --help
python scripts/make_contact_sheets.py --help
```

脚本会记录时间戳、清晰度、亮度、感知哈希和 SHA-256，便于复查某一张图的
来源。筛选结果不要直接视为真值，所有正式图片仍需在 X-AnyLabeling 中人工
确认。
