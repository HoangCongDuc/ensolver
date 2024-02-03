import argparse
import shutil
import os
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description="Convert coco formatted dataset into datasets with filename as text label.")
    parser.add_argument('input_json')
    parser.add_argument('input_imgdir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    json_path = args.input_json
    img_path = args.input_imgdir
    dst_dir = args.output_dir

    if os.path.exists(dst_dir):
        rm = input(f"{dst_dir} exists, remove [y/n]? ")
        if rm == 'y':
            shutil.rmtree(dst_dir)
        else:
            quit()
    os.makedirs(dst_dir)
    
    coco = COCO(json_path)
    for img_info in coco.imgs.values():
        img_id = img_info['id']
        anns = coco.imgToAnns[img_id]
        chars = []
        for ann in anns:
            catid = ann['category_id']
            char = coco.cats[catid]['name']
            chars.append(char)

        src_filename = img_info['file_name']
        src_path = os.path.join(img_path, src_filename)
        dst_filename = ''.join(chars) + '.png'
        dst_path = os.path.join(dst_dir, dst_filename)
        os.link(src=src_path, dst=dst_path)