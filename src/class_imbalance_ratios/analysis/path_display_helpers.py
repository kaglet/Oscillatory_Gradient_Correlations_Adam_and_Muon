from pathlib import Path

from src.class_imbalance_ratios.analysis.needed_paths import outputs_dir

def trim_path(path: Path) -> str:
    delim = "/"
    if "/" in str(path):
        delim = "/"
    elif "\\" in str(path): # Requires escaped characters for Windows never a single "\" in any str
        delim = "\\"
    else: 
        raise ValueError(f"Path does not contained required delimiters.\n Path given was: {path}")
    
    parts = str(path).split(delim)

    cut_idx = parts.index("outputs")
    new_path = "/".join(parts[cut_idx:])
    return new_path

if __name__ == "__main__":
    trimmed = trim_path(Path("C:/Users/motlh/coder_life/COS700/outputs/pretrained_muon_class_imbalance_ratios/multijob/R_20/run_0/val_desired_class_counts.pt"))
    print(trimmed)