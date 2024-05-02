import math
import os
import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms
import random

import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

# from nslt_dataset_all import NSLT as Dataset
from datasets.nslt_dataset_all import NSLT as Dataset
import cv2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


def load_rgb_frames_from_video(video_path, start=0, num=-1):
    vidcap = cv2.VideoCapture(video_path)
    # vidcap = cv2.VideoCapture('/home/dxli/Desktop/dm_256.mp4')

    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(num):
        success, img = vidcap.read()
        
        if not success:
            continue

        w, h, c = img.shape
        sc = 224 / w
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return torch.Tensor(np.asarray(frames, dtype=np.float32))


def run(init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        batch_size=3 * 15,
        save_model='',
        weights=None):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0

    top1_fp = np.zeros(num_classes, dtype=int)
    top1_tp = np.zeros(num_classes, dtype=int)

    top5_fp = np.zeros(num_classes, dtype=int)
    top5_tp = np.zeros(num_classes, dtype=int)

    top10_fp = np.zeros(num_classes, dtype=int)
    top10_tp = np.zeros(num_classes, dtype=int)
    # per_frame_logits = test_on_video(weights=weights, video_path='../../start_kit/raw_videos/00415.mp4', num_classes=2000)
    # print(per_frame_logits.shape)
    # print(per_frame_logits)
    predicted_result =[]
    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w
        print(inputs.shape)

        # Adjust input tensor size if the number of frames is not fixed
        num_frames = inputs.size(2)
        print(num_frames)
        if num_frames != 64:  # Assuming a fixed size of 64 frames
            # Pad or truncate frames to ensure the correct input size
            if num_frames < 64:
                pad = torch.zeros(1, 3, 64 - num_frames, 224, 224)
                inputs = torch.cat([inputs, pad], dim=2)
            else:
                inputs = inputs[:, :, :64, :, :]

        # Pass the inputs through the model
        per_frame_logits = i3d(inputs)

        # Process the logits to get the predicted label
        predictions = torch.max(per_frame_logits, dim=2)[0]
        predicted_label = torch.argmax(predictions[0]).item()
        print(predicted_label)

        predicted_result.append((video_id, predicted_label))

    return predicted_result
        # out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        # out_probs = np.sort(predictions.cpu().detach().numpy()[0])

        # if labels[0].item() in out_labels[-5:]:
        #     correct_5 += 1
        #     top5_tp[labels[0].item()] += 1
        # else:
        #     top5_fp[labels[0].item()] += 1
        # if labels[0].item() in out_labels[-10:]:
        #     correct_10 += 1
        #     top10_tp[labels[0].item()] += 1
        # else:
        #     top10_fp[labels[0].item()] += 1
        # if torch.argmax(predictions[0]).item() == labels[0].item():
        #     correct += 1
        #     top1_tp[labels[0].item()] += 1
        # else:
        #     top1_fp[labels[0].item()] += 1
        # print(video_id, float(correct) / len(dataloaders["test"]), float(correct_5) / len(dataloaders["test"]),
        #       float(correct_10) / len(dataloaders["test"]))
            
    # top1_per_class = np.nanmean(np.divide(top1_tp, (top1_tp + top1_fp)), axis=0)# this is to calculate the average per class accuracy which is called
    # top5_per_class = np.nanmean(np.divide(top5_tp, (top5_tp + top5_fp)), axis=0)
    # top10_per_class = np.nanmean(np.divide(top10_tp, (top10_tp + top10_fp)), axis=0)

    # # Handle NaN values
    # top1_per_class = np.nan_to_num(top1_per_class, nan=0.0)
    # top5_per_class = np.nan_to_num(top5_per_class, nan=0.0)
    # top10_per_class = np.nan_to_num(top10_per_class, nan=0.0)

    # # Print metrics
    # print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))
    # print_random_images_from_test_set(i3d, dataloaders)



        # per-class accuracy
    # top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    # top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    # top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    # print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))


def ensemble(mode, root, train_split, weights, num_classes):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    # test_transforms = transforms.Compose([])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0
    # confusion_matrix = np.zeros((num_classes,num_classes), dtype=np.int)

    top1_fp = np.zeros(num_classes, dtype=int)
    top1_tp = np.zeros(num_classes, dtype=int)

    top5_fp = np.zeros(num_classes, dtype=int)
    top5_tp = np.zeros(num_classes, dtype=int)

    top10_fp = np.zeros(num_classes, dtype=int)
    top10_tp = np.zeros(num_classes, dtype=int)

    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        t = inputs.size(2)
        num = 64
        if t > num:
            num_segments = math.floor(t / num)

            segments = []
            for k in range(num_segments):
                segments.append(inputs[:, :, k*num: (k+1)*num, :, :])

            segments = torch.cat(segments, dim=0)
            per_frame_logits = i3d(segments)

            predictions = torch.mean(per_frame_logits, dim=2)

            if predictions.shape[0] > 1:
                predictions = torch.mean(predictions, dim=0)

        else:
            per_frame_logits = i3d(inputs)
            predictions = torch.mean(per_frame_logits, dim=2)[0]

        out_labels = np.argsort(predictions.cpu().detach().numpy())

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        print(video_id, float(correct) / len(dataloaders["test"]), float(correct_5) / len(dataloaders["test"]),
              float(correct_10) / len(dataloaders["test"]))

    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))
    
def update_frame(frame):
    return frame
def print_video_with_labels(frames, true_label, predicted_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
    ax.set_axis_off()

    ims = []
    for frame in frames:
        im = ax.imshow(frame.permute(1, 2, 0))
        ims.append([im])

    ani = FuncAnimation(fig, update_frame, frames=ims, blit=True, repeat=False)
    plt.show()
    ani = FuncAnimation(fig, update_frame, frames=ims, blit=True, repeat=False)
    plt.show()

def predict_label(i3d, inputs):
    per_frame_logits = i3d(inputs)
    predictions = torch.max(per_frame_logits, dim=2)[0]
    predicted_label = torch.argmax(predictions[0]).item()
    print(f"Predicted label: {predicted_label}")
    return predicted_label

def print_random_images_from_test_set(model, dataloaders, root, num_videos=2):
    count = 0
    for data in dataloaders['test']:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w
        # Your code to process and print images here
        print(f"Video ID: {video_id}")
        predicted_label = predict_label(model, inputs)
        print_video_with_labels(video_id, labels[0], predicted_label, root)
        # Print additional information as needed
        count += 1
        if count >= num_videos:
            break
    
def test_on_camera(weights, num_classes):
    cap = cv2.VideoCapture(0)
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights)) 
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = (frame / 255.) * 2 - 1
        frame = np.asarray([frame]).transpose([0, 3, 1, 2])
        frame = torch.Tensor(frame).cuda()
        frame = frame.unsqueeze(0)  # Add this line
        print(frame.shape)

        predicted_label = predict_label(i3d, frame)
        print(predicted_label)
        
        print(predicted_label)

        cv2.imshow('frame', frame.cpu().numpy()[0].transpose(1, 2, 0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_frames_from_video(video_path, target_size=(224, 224)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.resize(frame, target_size)  # Resize to target size
        frame = (frame / 255.0) * 2 - 1  # Normalize to [-1, 1]
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"Warning: No frames read from video '{video_path}'")
        return []

    frames = np.stack(frames, axis=0)  # Stack frames along a new axis
    return frames

def test_on_video(weights, video_path, num_classes, segment_length=64):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    frames = load_frames_from_video(video_path)
    print(frames.shape)  # (num_frames, height, width, channels)

    num_frames = frames.shape[0]
    per_frame_logits_list = []

    # Process the video in segments of segment_length frames
    for start_idx in range(0, num_frames, segment_length):
        end_idx = min(start_idx + segment_length, num_frames)
        segment_frames = frames[start_idx:end_idx]

        # Reshape frames to (num_frames, channels, height, width)
        segment_frames = segment_frames.transpose(0, 3, 1, 2)

        # Add a batch dimension
        segment_frames_tensor = torch.from_numpy(segment_frames).unsqueeze(0).cuda()

        with torch.no_grad():
            per_frame_logits = i3d(segment_frames_tensor)

        per_frame_logits_list.append(per_frame_logits)

    # Concatenate the logits from all segments
    per_frame_logits = torch.cat(per_frame_logits_list, dim=2)

    return per_frame_logits
    






def run_on_tensor(weights, ip_tensor, num_classes):
    i3d = InceptionI3d(400, in_channels=3)
    # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    t = ip_tensor.shape[2]
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)

    predictions = F.upsample(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])

    arr = predictions.cpu().detach().numpy()[0,:,0].T

    plt.plot(range(len(arr)), F.softmax(torch.from_numpy(arr), dim=0).numpy())
    plt.show()

    return out_labels


def get_slide_windows(frames, window_size, stride=1):
    indices = torch.arange(0, frames.shape[0])
    window_indices = indices.unfold(0, window_size, stride)

    return frames[window_indices, :, :, :].transpose(1, 2)


# if __name__ == '__main__':
#     # ================== test i3d on a dataset ==============
#     # need to add argparse
#     mode = 'rgb'
#     num_classes = 2000
#     save_model = './checkpoints/'

#     # root = '../../start_kit/raw_videos'
#     root  = '../../start_kit/raw_videos/02951.mp4'
#     # root = '../../videos'

#     train_split = 'preprocess/nslt_{}.json'.format(num_classes)
#     weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
#     # weights = None
#     predicted_result = run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
#     for video_id, predicted_label in predicted_result:
#         print(f"Video ID: {video_id}, Predicted Label: {predicted_label}")
        
if __name__ == '__main__':
    # Define the path to the video file
    video_path ='../../start_kit/raw_videos/02951.mp4'

    # Define the mode (rgb or flow), number of classes, and model weights
    mode = 'rgb'
    num_classes = 2000
    weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'

    per_frame_logits = test_on_video(weights, video_path, num_classes)

    # Process the logits to get the predicted label
    predictions = torch.max(per_frame_logits, dim=2)[0]
    predicted_label = torch.argmax(predictions[0]).item()

    print(f"Predicted label for the video: {predicted_label}")
 