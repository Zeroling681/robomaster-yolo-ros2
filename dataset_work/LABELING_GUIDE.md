# X-AnyLabeling 标注说明

## 启动

在 PowerShell 中运行：

```powershell
Set-Location F:\PycharmProjects\robomaster
powershell -ExecutionPolicy Bypass -File .\scripts\open_anylabeling.ps1
```

正式待标注图片目录为：

`F:\PycharmProjects\robomaster\dataset_work\anylabeling_dataset\images`

不要使用 `rejected_frames` 中的图片。

程序已载入预训练 YOLO 生成的辅助框，但辅助框不是最终真值。优先检查：

`anylabeling_dataset\PRELABEL_REVIEW_PRIORITY.csv`

其中列出了40张主目标缺失图片：34张完全无辅助框，6张只有错误类别框。
其余图片也必须逐张检查，修正框的位置、漏掉的背景目标和错误类别。

## 类别

类别文件为 `anylabeling_dataset\classes.txt`，顺序必须保持不变：

```text
mouse
cup
```

- `mouse`：类别编号 0。
- `cup`：类别编号 1；本项目将塑料饮料瓶、运动水杯和保温杯统一归为该类。

## 画框规则

1. 使用矩形框工具，框紧贴物体可见外轮廓。
2. 每张图片中所有可见的鼠标和杯子都必须标注，包括背景里的小目标。
3. 手、键盘、电脑、线缆和桌面不标注。
4. 轻度遮挡或处于画面边缘的目标仍需标注，框住其可见部分。
5. 标签只使用小写英文 `mouse`、`cup`，不要创建同义标签。
6. 每完成一张立即保存，并在进入下一张前确认类别和框的位置。
7. 如果已有辅助框，先确认类别；鼠标上的蓝色 `cup` 框、杯盖上的绿色
   `mouse` 框都属于错误框，应删除或改成正确类别。

## 导出

完成全部图片后，选择 `Export Annotations` → `YOLO Annotations` →
目标检测，并使用 `classes.txt` 作为类别配置。YOLO 标签应导出到：

`F:\PycharmProjects\robomaster\dataset_work\anylabeling_dataset\labels`

导出后，每张图片都应存在同名 `.txt`。没有目标的负样本应保留同名空文件。

