$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$executable = 'F:\Tools\X-AnyLabeling\X-AnyLabeling-v4.0.3-CPU.exe'
$images = 'F:\PycharmProjects\robomaster\dataset_work\anylabeling_dataset\images'
$annotations = 'F:\PycharmProjects\robomaster\dataset_work\anylabeling_dataset\annotations_xlabel'

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "找不到 X-AnyLabeling: $executable"
}
if (-not (Test-Path -LiteralPath $images -PathType Container)) {
    throw "找不到图片目录: $images"
}

New-Item -ItemType Directory -Force -Path $annotations | Out-Null
Start-Process -FilePath $executable -ArgumentList @(
    '--filename', $images,
    '--output', $annotations
)

Write-Host "X-AnyLabeling started. Images: $images"
Write-Host "Native annotations: $annotations"
