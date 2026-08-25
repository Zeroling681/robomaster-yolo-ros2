# 视频抽帧数据集状态

## 正式数据

- 图片总数：188张。
- `mouse`：93张主场景图片。
- `cup`：95张主场景图片。
- 来源：6个视频，每类3个实物视频。
- 精确重复图片：0张。
- 正式图片：`anylabeling_dataset\images`。
- X-AnyLabeling 原生标注：`anylabeling_dataset\annotations_xlabel`。
- 最终 YOLO 标签：`anylabeling_dataset\labels`，人工复核并导出前为空。

4张目标只剩极小边角的图片以及一次无效随机跳转批次已移动到
`rejected_frames`，不得用于训练。

## 辅助标注

预训练 YOLO 为154张图片生成了候选框，34张无框。按每张图片的主场景类别
检查后，共40张需要优先复核，清单位于：

`anylabeling_dataset\PRELABEL_REVIEW_PRIORITY.csv`

辅助标注会漏检和误分类，不能直接作为最终数据集。所有188张都需要在
X-AnyLabeling 中人工确认。

## 后续划分注意事项

这些图片来自视频，相邻画面高度相关。完成标注后不能直接逐图片随机划分，
否则训练集和测试集会出现近似帧，导致指标虚高。应按来源视频和连续时间段
分组划分，并另外拍摄不同背景、光照以及两类物体同时出现的独立测试场景。

