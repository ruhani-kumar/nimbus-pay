import os
import sys
import subprocess
from importlib.metadata import version, PackageNotFoundError

IN_COLAB = "google.colab" in sys.modules

PINS = {
    "transformers": "5.22.1",
    "peft": "0.21.2",
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
}
EXPECTED_TORCH = "2.14.1"


def installed(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


if IN_COLAB:
    missing = [f"{p}=={v}" for p, v in PINS.items() if installed(p) != v]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

print("colab:", IN_COLAB)
print(f"torch=={installed('torch')}", "" if (installed("torch") or "").startswith(EXPECTED_TORCH) else f"(expected {EXPECTED_TORCH})")
for pkg, want in PINS.items():
    have = installed(pkg)
    print(f"{pkg}=={have}", "" if have == want else f"(expected {want})")
