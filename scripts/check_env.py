from __future__ import annotations

import platform

import cv2
import numpy as np
import rclpy
import torch
import ultralytics
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 未检测到 CUDA GPU")

    tensor = torch.rand((1024, 1024), device="cuda")
    checksum = float((tensor @ tensor).mean().item())
    torch.cuda.synchronize()

    rclpy.init(args=None)
    rclpy.shutdown()

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"OpenCV: {cv2.__version__}; NumPy: {np.__version__}")
    print(f"ROS 2 Python: OK ({Detection2DArray.__name__}, {CvBridge.__name__})")
    print(f"CUDA calculation checksum: {checksum:.6f}")
    print("ENVIRONMENT_CHECK=PASS")


if __name__ == "__main__":
    main()

