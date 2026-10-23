
from pathlib import Path

from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_s_outputs_dir, cls_imb_l_outputs_dir
from src.class_imbalance_ratios.analysis.post_proc_convert import delete_all_non_extension, delete_all_parquet_extras, delete_all_pt_extension
from src.class_imbalance_ratios.analysis.post_proc_convert import convert_all_pt_metrics_to_parquet, convert_pt_class_counts_to_parquet
import logging
import os 

def clean_non_core(dir: Path):
    """
    These can be run safely in case of mistakes in 
    downstream extras based metrics. 
    """
    delete_all_pt_extension(outputs_dir=dir)
    delete_all_parquet_extras(outputs_dir=dir)

    log_file = Path(os.getcwd()) / "src/log_analysis.log"
    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    print("=========== DELETING ALL PARQUET FILES EXPECTED BUT MISSING .PARQUET ===========")
    delete_all_non_extension(outputs_dir=dir)

    print("=========== CONVERTING ALL .PT files to .PARQUET ===========")
    convert_all_pt_metrics_to_parquet(outputs_dir=dir) # Took 30m 50.7s on CIF10 cls_imb locally

if __name__ == "__main__":
    """ 
    Ensure core metrics are kept in .parquet format while 
    non-core e.g. extras, or non-parquet files are cleaned. 
    """
    clean_non_core(cls_imb_l_outputs_dir)