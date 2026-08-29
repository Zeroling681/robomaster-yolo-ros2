$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$executable = 'F:\Tools\X-AnyLabeling\X-AnyLabeling-v4.0.3-CPU.exe'
$dataset = 'F:\PycharmProjects\robomaster\dataset_work\audit_dataset'
$images = Join-Path $dataset 'images'
$annotations = Join-Path $dataset 'annotations'

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "找不到 X-AnyLabeling: $executable"
}
if (-not (Test-Path -LiteralPath $images -PathType Container)) {
    throw "找不到审计图片目录: $images"
}

New-Item -ItemType Directory -Force -Path $annotations | Out-Null
Start-Process -FilePath $executable -ArgumentList @(
    '--filename', $images,
    '--output', $annotations
)

Write-Host "X-AnyLabeling 已启动。"
Write-Host "图片目录: $images"
Write-Host "标注目录: $annotations"
