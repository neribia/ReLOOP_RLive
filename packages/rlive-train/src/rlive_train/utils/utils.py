import sys
import torch

def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available torch device.

    Args:
        prefer: "auto", "cuda", "mps", or "cpu".

    Returns:
        torch.device: Selected device.
    """
    prefer = prefer.lower()

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if prefer == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # auto: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

if __name__ == "__main__":
    print(get_device())