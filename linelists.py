from tqdm import tqdm
from src.utils.tsv_file_ops import tsv_writer, tsv_reader, generate_linelist_file
import json 
import argparse
    

def generate_linelist_file(split="public_test", ignore_attrs=()):
    # generate a list of image that has labels
    # images with only ignore labels are not selected. 
    line_list = []
    rows = tsv_reader(f"datasets/VATEX/{split}.label.tsv")
    for i, row in tqdm(enumerate(rows)):
        labels = json.loads(row[1])
        if labels:
            if ignore_attrs and all([any([lab[attr] for attr in ignore_attrs if attr in lab]) \
                                for lab in labels]):
                continue
            line_list.append([i])

    save_file = f"datasets/VATEX/{split}.linelist.tsv"
    tsv_writer(line_list, save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="train/val/private_test/public_test", default="public_test")
    args = parser.parse_args()
    generate_linelist_file({args.split})