from pathlib import Path

__all__ = ["outputs_dir"]

outputs_dir = Path(__file__).parent.parent.parent.parent / "outputs"

if __name__ == "__main__":
    print("Outputs dir verified as: ", outputs_dir) 