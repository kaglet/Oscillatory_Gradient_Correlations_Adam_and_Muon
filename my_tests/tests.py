import argparse
from pathlib import Path
import sys

def get_sum_of_num(num1,num2,num3):
  return(int(num1)+int(num2)+int(num3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that adds 3 numbers from CMD"
    )
    parser.add_argument("--num1", required=True, type=int, default=1)
    parser.add_argument("--num2", required=True, type=int, default=4)
    parser.add_argument("--num3", required=True, type=int, default=6, help="Enter third numbe")
    args = parser.parse_args()

    num1 = args.num1
    num2 = args.num2
    num3 = args.num3

    print(get_sum_of_num(num1, num2, num3))

