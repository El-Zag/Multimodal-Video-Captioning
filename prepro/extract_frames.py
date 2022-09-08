import os
from os.path import join
from tqdm import tqdm
import multiprocessing as mp
import subprocess
import datetime
import pandas as pd

def get_video_duration(video_file):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", video_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def extract_frame_from_video(video_path, save_frame_path, fps=1, num_frames=-1,
                             start_ts=-1, end_ts=-1,
                             suppress_msg=False, other_args="", overwrite=True):
    """Uniformly split a video into clips of length {clip_len}.
    i.e., in the case of clip_len=60, the clips will be 00:00:00-00:01:00, 00:01:00-00:02:00, etc, ...

    Note that we drop the first (usually opening remark, etc) and last (ask for subscription, etc) clip.

    Args:
        video_path:
        save_frame_path:
        fps: frame_per_second, default 1
        suppress_msg:
        other_args: str, other ffmpeg args, such as re-scale to 720p with '-vf scale=-1:720'

    Returns:

    """
    extra_args = " -hide_banner -loglevel panic " if suppress_msg else ""
    extra_args += " -y " if overwrite else ""
    youtube_id = video_path.split('/')[-1].split('.')[0] ## CHECK THAT
    start_ts = pd_time.loc[youtube_id, "Start time"] ## CHECK THAT 
    if youtube_id == "baEoQJbrqP0" :
        print(youtube_id)
        suppress_msg = False
    end_ts = pd_time.loc[youtube_id,"End time"] ## CHECK THAT 
    try :
        assert start_ts != -1 and end_ts != -1
    except :
        print(youtube_id, start_ts, end_ts)
    if start_ts != -1 and end_ts != -1:
        #start_ts_str = str(datetime.timedelta(seconds=start_ts))
        #end_ts_str = str(datetime.timedelta(seconds=end_ts))
        #duration = str(datetime.timedelta(seconds=(end_ts - start_ts)))
        # print(start_ts, end_ts, duration)
        #extra_args += f"-ss {start_ts_str} -t {duration} "
        extra_args += f"-ss {start_ts} -to {end_ts} "
    # extra_args2 = " -vf scale=720:-2 "
    # -preset veryfast:  (upgrade to latest ffmpeg if error)
    # https://superuser.com/questions/490683/cheat-sheets-and-presets-settings-that-actually-work-with-ffmpeg-1-0
    if num_frames <= 0 :
        split_cmd_template = "ffmpeg {extra} -i {video_path} -vf fps={fps} {output_frame_path}%06d.jpg"
        
        cur_split_cmd = split_cmd_template.format(
            extra=extra_args, video_path=video_path, fps=fps, output_frame_path=save_frame_path)
    else:
        # get duration of the video
        if start_ts != -1 and end_ts != -1:
            duration = get_sec(end_ts) - get_sec(start_ts)
        else:
            duration = get_video_duration(video_path)
        if duration <= 0:
            duration = 10
            print(video_path)
        frame_rate = num_frames/duration
        # if not suppress_msg:
        #     print(duration, frame_rate, num_frames)
        output_exists = True
        for frame_idx in range(num_frames):
            if not os.path.exists(f"{save_frame_path}{(frame_idx+1):04d}.jpg"):
                #print(f"{save_frame_path}{(frame_idx+1):04d}.jpg does not exist")
                output_exists = False
                save_frame_path = save_frame_path.replace(f"{num_frames}frames_test_value", f"{num_frames}frames_test_value_debug")
                break
                
        if output_exists:
            print(f"frames {save_frame_path} already exist")
            return
        split_cmd_template = "ffmpeg {extra} -i {video_path} -vf fps={frame_rate} {output_frame_path}%04d.jpg"

        cur_split_cmd = split_cmd_template.format(
            extra=extra_args, video_path=video_path, frame_rate=frame_rate, output_frame_path=save_frame_path)
        if not suppress_msg:
            print(cur_split_cmd)
    try:
        _ = subprocess.run(cur_split_cmd.split(), stdout=subprocess.PIPE)
    except Exception as e:
        print(f"Error returned by ffmpeg cmd {e}")


COMMON_VIDEO_ETX = set([
    ".webm", ".mpg", ".mpeg", ".mpv", ".ogg",
    ".mp4", ".m4p", ".mpv", ".avi", ".wmv", ".qt",
    ".mov", ".flv", ".swf"])


def extract_frame(video_file_path, save_dir, fps, num_frames, debug=False, corrupt_files=[]):
    filename = os.path.basename(video_file_path)
    vid, _ = os.path.splitext(filename)
    frame_name = f"{vid}_frame"
    frame_save_path = join(save_dir, frame_name)

    if (video_file_path not in corrupt_files and len(corrupt_files)):
        print(f"skipping {video_file_path}")
        return
    if len(corrupt_files):
        print(f"exracting frames for {video_file_path}")
    launch_extract = True
    if launch_extract:
        os.makedirs(save_dir, exist_ok=True)
        # scale=width:height
        extract_frame_from_video(video_file_path, frame_save_path, fps=fps, num_frames=num_frames,
                                 suppress_msg=True, other_args="")


def load_tsv_to_mem(tsv_file, sep='\t' ):
    data = []
    with open(tsv_file, 'r') as fp:
        for _, line in enumerate(fp):
            data.append([x.strip() for x in line.split(sep)])
    return data


def extract_all_frames(split, save_dir, fps, num_frames,
                    corrupt_files, num_workers, debug=False):

    raw_video_info = load_tsv_to_mem(f'datasets/VATEX/{split}.img.tsv')
    videoFiles = []
    for _, line_item in enumerate(raw_video_info):
        input_file = line_item[0]
        #input_file = input_file.replace('datasets','_datasets')
        if os.path.isfile(input_file):
            videoFiles.append(input_file)


    if debug:
        videoFiles = videoFiles[:1]

    if num_workers > 0:
        from functools import partial
        extract_frame_partial = partial(
            extract_frame, fps=fps,
            save_dir=save_dir, debug=debug, corrupt_files=corrupt_files,
            num_frames=num_frames)

        with mp.Pool(num_workers) as pool, tqdm(total=len(videoFiles)) as pbar:
            for idx, _ in enumerate(
                    pool.imap_unordered(
                        extract_frame_partial, videoFiles, chunksize=8)):
                pbar.update(1)
    else:
        for idx, d in tqdm(enumerate(videoFiles),
                           total=len(videoFiles), desc="extracting frames from video"):
            extract_frame(d, save_dir, fps=fps, debug=debug)
            if debug and idx >= 10:
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="train/val/private_test/public_test", default="train")
    parser.add_argument("--save_dir", type=str, help="save frame dir ", default="datasets/VATEX/")
    parser.add_argument("--fps", type=str, default="1")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--corrupt_file_path", type=str, default="",
                        help="dir saving output videos")
    args = parser.parse_args()
    args.save_dir = args.save_dir + str(args.num_frames) + 'frames'
    corrupt_files = []
    if os.path.exists(args.corrupt_file_path):
        with open(args.corrupt_file_path) as f:
            lines = f.readlines()
            for ll in lines:
                corrupt_files.append(ll.strip("\n"))

    if args.split=="train" :
        pkl = "pickles/videos_to_dl.pkl"
    elif args.split=="val":
        pkl = "pickles/validation_to_dl.pkl"
    elif args.split=="private_test" or args.split=="public_test":
        pkl = "pickles/test_to_dl.pkl"
    
    pd_time = pd.read_pickle(pkl).set_index('videoID')

    extract_all_frames(args.split, args.save_dir, args.fps,
                       args.num_frames, corrupt_files,
                       num_workers=args.num_workers, debug=args.debug)