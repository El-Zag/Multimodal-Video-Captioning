from pathlib import Path
import os
import json
from tqdm import tqdm
import argparse

def prepare_tsv(tsv, dir, file, ignore):
    with open(f"datasets/VATEX/_og.{tsv}.{file}.tsv", "r") as f:
        lines = f.readlines()
    to_keep = []
    for idx, line in enumerate(lines):
        id = Path([x.strip() for x in line.split('\t')][0]).stem
        if id in ignore:
            continue
        if os.path.exists(f"/mnt/video-galvatron-data/users/eloisez/youtube_dl/Audio_dl/{dir}/{id}.wav"):
            to_keep.append(idx) 

    with open(f"datasets/VATEX/{tsv}.{file}.tsv", "w") as f:
        for idx, line in enumerate(lines):
            if idx in to_keep :
                f.write(line)
    
    return len(to_keep)


def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

def generate_caption_linelist_file(split):
    num_captions = []
    for row in tsv_reader(f"datasets/VATEX/{split}.caption.tsv"):
        num_captions.append(len(json.loads(row[1])))
    cap_linelist = ['\t'.join([str(img_idx), str(cap_idx)]) 
            for img_idx in range(len(num_captions)) 
            for cap_idx in range(num_captions[img_idx])
    ]
    save_file = f"datasets/VATEX/{split}.caption.linelist.tsv"
    with open(save_file, 'w') as f:
        f.write('\n'.join(cap_linelist))
    return save_file

def generate_json_coco(tsv, dir, n):
    data = tsv_reader(f"datasets/VATEX/{tsv}.caption.tsv")
    good_videos, good_videos10 = [], []
    for row in data :
        good_videos10 += 10 * [row[0]]
        good_videos.append(row[0])
    
    assert len(good_videos)*10 == len(good_videos10)
    assert len(good_videos) == n

    counter = 0
    with open(f"datasets/VATEX/_og.{tsv}.caption_coco_format.json", "r") as f :
        data = json.load(f)
    ### ANNOTATION PART OF THE COCO FORMAT
    good_annotations = []
    old_n = len(data['annotations'])
    print(f"Old number of captions {old_n}")
    for element in tqdm(data['annotations']):
        youtube_id = Path(element['image_id']).stem[:11]
        if not os.path.isfile(f"/mnt/video-galvatron-data/users/eloisez/youtube_dl/Video_dl/{dir}/{youtube_id}.mp4"):
            continue
        if not os.path.isfile(f"/mnt/video-galvatron-data/users/eloisez/youtube_dl/Audio_dl/{dir}/{youtube_id}.wav"):
            continue
        if not good_videos10[counter] == f'rawvideos/{dir}/{youtube_id}.mp4':
            continue
        else :
            element['image_id'] = f'rawvideos/{dir}/{youtube_id}.mp4'
            element['id'] = counter
            counter +=1
            good_annotations.append(element)
    data['annotations'] = good_annotations
    new_n = len(data['annotations'])
    assert new_n == len(good_videos10)

    ### ANNOTATION PART OF THE COCO FORMAT
    assert len(data['images'])*10 == old_n
    counter = 0
    good_images = []
    for element in tqdm(data['images']):
        youtube_id = Path(element['id']).stem[:11]
        
        if not os.path.isfile(f"/mnt/video-galvatron-data/users/eloisez/youtube_dl/Video_dl/{dir}/{youtube_id}.mp4"):
            continue
        if not os.path.isfile(f"/mnt/video-galvatron-data/users/eloisez/youtube_dl/Audio_dl/{dir}/{youtube_id}.wav"):
            continue
        if not good_videos[counter] == f'rawvideos/{dir}/{youtube_id}.mp4':
            continue
        else :
            element['id'] = f'rawvideos/{dir}/{youtube_id}.mp4'
            element['file_name'] = element['id']
            good_images.append(element)
            counter += 1
    
    data['images'] = good_images
    assert len(data['images']) == n

    with open(f"datasets/VATEX/{tsv}.caption_coco_format.json", "w") as f :
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    
    ignore = ["M7wK6IucSEo", "U7LC5P8VDZw", "8T759EZNHRI",
            "p0eXF9-ZLdE", "3W4-akyf6wA", "ohVebc-gLHA", "Bk0i4M38dLk", 
            "tREQGo8rNoA", "98kXym5hBXI", "FJOQAoGBFkU", "DoSnLd9ZK4o", 
            "ym3gSCDjQJI", "_cueJ0tT5UQ", "CPI3Sy0Ic54", "Q7UqFn8xMv0",
            "c3DAquBQ2dg", "baEoQJbrqP0", "PictzbnzSPA", "beP7m1yQpSQ",
            "5mBqtX4KrGk", "ODsFDvcEXh0", "SCIQzIfzOmw"]

    tsvs = ["train", "val", "public_test"]
    dirs = ["training", "validation", "test"]
    dock = input("Are you in the docker ?")

    if dock =="n" :
    ### GENERATE ALL THE NEW TSVs
        print("--------- NOT IN DOCKER ---------")
        print("--------- Generating tsv files ---------")
        n_files= dict()
        for (tsv, dir) in zip(tsvs, dirs):
            print(f"For {tsv}:")
            check = []
            for file in ['img', 'caption', 'label']:
                print(f"Generating {file} file...")
                check.append(prepare_tsv(tsv, dir, file, ignore))
            assert check[0] == check[1], "Not the same number of files between img and caption"
            assert check[1] == check[2], "Not the same number of files between caption and label"
            n_files[tsv] = check[0]
        
        print(f"Number of files per split {n_files}")

        #n_files = {'train': 21884, 'val': 1364, 'public_test': 5093}
        print("--------- Deleting old lineidx files ---------")
        ### DELETE THE OLD LINEIDX
        counter = 0
        for file in os.listdir("datasets/VATEX"):
            if file.endswith('.lineidx'):
                os.remove(f"datasets/VATEX/{file}")
                counter += 1
        print(f"Deleted {counter} lineidx files")
    
        print("--------- Generating json coco caption files ---------")
        for (tsv, dir) in zip(tsvs, dirs):
            print(f"For {tsv}...")
            generate_json_coco(tsv, dir, n_files[tsv])
        
        print("--------- Now go to docker to do the rest ---------")
    
    if dock =="y":
        print("--------- IN DOCKER ---------")
        from linelists import generate_linelist_file
        from prepro.create_image_frame_tsv import main

        print("--------- Generating new linelists files ---------")
        for tsv in tsvs : 
            ### GENERATE NEW CAPTION LINELISTS
            generate_caption_linelist_file(tsv)
            ### GENERATE LABEL LINELIST
            generate_linelist_file(tsv)

        print("--------- Generating image frame tsv ---------")
        for tsv in tsvs :
            print(f"For {tsv}...")
            main(tsv=tsv)
    



    
    