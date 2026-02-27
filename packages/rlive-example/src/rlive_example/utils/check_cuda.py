import torch


def check_cuda() -> None:
    print("=" * 50)
    print("CUDA Diagnostics")
    print("=" * 50)

    print(f"PyTorch version:       {torch.__version__}")
    print(f"CUDA available:        {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:          {torch.version.cuda}")
        print(f"cuDNN version:         {torch.backends.cudnn.version()}")
        print(f"Device count:          {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024 ** 3)
            print(f"  [{i}] {props.name} ({vram:.1f} GB)")
        print(f"Current device:        {torch.cuda.current_device()}")
        print(f"Current device name:   {torch.cuda.get_device_name()}")

        # Quick tensor test
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        print(f"Tensor on CUDA test:   {t.device} ✓")
    else:
        print()
        print("⚠ CUDA is NOT available.")
        print("  Possible causes:")
        print("  - PyTorch was installed without CUDA support (CPU-only)")
        print("  - No NVIDIA GPU detected")
        print("  - NVIDIA drivers are outdated")
        print()
        print("  To install PyTorch with CUDA, run:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu126")

    print("=" * 50)


if __name__ == "__main__":
    check_cuda()