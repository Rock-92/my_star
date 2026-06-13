from __future__ import annotations

import json
import platform
import sys

import torch


def main() -> None:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "known_privateuse1_backend": torch._C._get_privateuse1_backend_name(),
    }
    if torch.cuda.is_available():
        report["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
        try:
            device = torch.device("cuda")
            left = torch.randn(256, 256, device=device)
            right = torch.randn(256, 256, device=device)
            result = left @ right
            report["cuda_tensor_smoke"] = {
                "ok": True,
                "device": str(result.device),
                "mean": float(result.mean().cpu()),
            }
        except Exception as error:
            report["cuda_tensor_smoke"] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    for module_name in ("torch_npu", "torch_mlu", "torch_dipu", "torch_gcu"):
        try:
            module = __import__(module_name)
            report[module_name] = getattr(module, "__version__", "installed")
        except Exception as error:
            report[module_name] = f"unavailable: {type(error).__name__}"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
