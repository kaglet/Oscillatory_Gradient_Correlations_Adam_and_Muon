from pathlib import Path

__all__ = ["cls_imb_outputs_dir"]

cls_imb_outputs_dir = Path(__file__).parent.parent.parent.parent / "outputs/outputs_cls_imb_s"

if __name__ == "__main__":
    print("Cls imb outputs dir verified as: ", cls_imb_outputs_dir) 