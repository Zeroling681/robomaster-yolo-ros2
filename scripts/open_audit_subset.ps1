param(
    [ValidateSet('marked', 'missing')]
    [string]$Subset = 'missing'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$executable = 'F:\Tools\X-AnyLabeling\X-AnyLabeling-v4.0.3-CPU.exe'
$datasetRoot = 'F:\PycharmProjects\robomaster\dataset_work\audit_dataset'
$viewName = if ($Subset -eq 'marked') { 'review_with_boxes' } else { 'review_missing' }
$view = Join-Path $datasetRoot $viewName
$images = Join-Path $view 'images'
$annotations = Join-Path $view 'annotations'

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw "找不到 X-AnyLabeling: $executable" }
if (-not (Test-Path -LiteralPath $images -PathType Container)) { throw "找不到筛选目录: $images" }
Start-Process -FilePath $executable -ArgumentList @('--filename', $images, '--output', $annotations)
Write-Host "X-AnyLabeling 已启动: $viewName"
Write-Host "图片: $images"
Write-Host "标注: $annotations"
