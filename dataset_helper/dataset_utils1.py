import json
import os
import os.path
import cv2
import numpy as np
import torch
import torch.utils.data as data_utl

def load_rgb(vid_root, vid, start, num):
    video_root_dir = vid_root
    video_id = vid
    start_frame = start
    frames_to_load = num

    video_path = os.path.join(video_root_dir, video_id + '.mp4')
    cap = cv2.VideoCapture(video_path)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(frames_to_load):
        success, img = cap.read()
        if not success or img is None:
            break

        height, width, _ = img.shape
        if width < 226 or height < 226:
            min_dim = min(width, height)
            target_dim = 226.0
            scale = target_dim / min_dim
            img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        img = (img / 255.0) * 2.0 - 1.0
        frames.append(img)

    cap.release()

    if not frames:
        return np.asarray([], dtype=np.float32)


    return np.asarray(frames, dtype=np.float32)

def load_flow(image_dir, vid, start, num):
    frame_root_dir = image_dir
    video_id = vid
    start_frame = start
    frames_to_load = num

    frames = []
    for i in range(start_frame, start_frame + frames_to_load):
        frame_idx_str = str(i).zfill(6)
        flow_x_path = os.path.join(frame_root_dir, video_id, f"{video_id}-{frame_idx_str}x.jpg")
        flow_y_path = os.path.join(frame_root_dir, video_id, f"{video_id}-{frame_idx_str}y.jpg")

        imgx = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)

        ''' 
        if imgx is None or imgy is None:
            print(f"Warning: Could not read flow frame pair for index {i} in {video_id}")
            continue
        '''

        height, width = imgx.shape

        if width < 224 or height < 224:
            min_dim = min(width, height)
            target_dim = 224.0
            scale = target_dim / min_dim
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        imgx = (imgx / 255.0) * 2.0 - 1.0
        imgy = (imgy / 255.0) * 2.0 - 1.0

        img = np.stack([imgx, imgy], axis=-1)
        frames.append(img)

    if not frames:
        return np.asarray([], dtype=np.float32)

    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes):
    split_filepath = split_file
    video_root_dir = root
    dataset_mode = mode

    dataset_list = []
    with open(split_filepath, 'r') as f:
        data = json.load(f)

    for video_id in data.keys():
        if data[video_id]['subset'] != "test":
            continue

        video_path = os.path.join(video_root_dir, video_id + '.mp4')
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path} to get frame count.")
            continue
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        dataset_list.append((video_id, data[video_id]['action'][0], 0, num_frames, video_id))

    print(len(dataset_list))
    return dataset_list


def get_num_class(split_file):
    split_filepath = split_file
    class_set = set()

    with open(split_filepath, 'r') as f:
        content = json.load(f)

    for video_id in content.keys():
        class_id = content[video_id]['action'][0]
        class_set.add(class_id)

    return len(class_set)

def tensorize(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class Utils(data_utl.Dataset):
    def __init__(self, split_file, split, root, mode, transforms=None):
        self.num_classes = get_num_class(split_file)
        _root = root['word'] if isinstance(root, dict) else root
        self.data = make_dataset(split_file, split, _root, mode, self.num_classes)
        self.transforms = transforms
        self.mode = mode
        self.root = _root

    def __getitem__(self, index):
        vid, label, start_f, frame_count, _ = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb(self.root, vid, start_f, frame_count)
        else:
             imgs = load_flow(self.root, vid, start_f, frame_count)

        if imgs.size == 0 :
            print(f"Warning: No frames loaded for video {vid}, index {index}. Returning empty tensor.")
            channels = 3 if self.mode == 'rgb' else 2

            return torch.empty((channels, 0, 0, 0)), label, vid


        if self.transforms:
            imgs = self.transforms(imgs)


        ret_img = tensorize(imgs)

        if ret_img.shape[1] == 0:
             print(f"Warning: Returning tensor with 0 frames after transforms for video {vid}, index {index}.")

        return ret_img, label, vid

    def __len__(self):
        return len(self.data)