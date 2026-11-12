import torch
import transformers


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


set_seed()

print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    NATIVE_BF16 = props.major >= 8
    print("gpu:", props.name, f"{props.total_memory / 1e9:.1f} GB", f"| compute capability {props.major}.{props.minor}",
          "| native bf16:", NATIVE_BF16, "| training dtype:", "bf16" if NATIVE_BF16 else "fp16")
else:
    print("gpu: none")
