from pathlib import Path

__all__ = ["cls_imb_s_outputs_dir", "cls_imb_l_outputs_dir"]

cls_imb_s_outputs_dir = Path(__file__).parent.parent.parent.parent / "outputs/cls_imb_s"
cls_imb_l_outputs_dir = Path(__file__).parent.parent.parent.parent / "outputs/cls_imb_l"

if __name__ == "__main__":
    print("Cls imb small outputs dir verified as: ", cls_imb_s_outputs_dir) 
    print("Cls imb large outputs dir verified as: ", cls_imb_l_outputs_dir) 