import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import math
import numpy as np

import videotransforms
from pytorch_i3d import InceptionI3d
# from .pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import load_rgb_frames_from_video, video_to_tensor, load_rgb_frames_from_live_feed
from decoder import get_gloss
import dask
import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
from chatgpt.prompt import chat_with_claude


NUM_CLASSES = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = 'checkpoints/nslt_2000_065846_0.447803.pt'
# weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
# VIDEO_ROOT = '../../start_kit/raw_videos/'
VIDEO_ROOT = '../../videos'


def pad(imgs, total_frames=64):
    if imgs.shape[0] < total_frames:
        print(imgs.shape[0])
        num_padding = total_frames - imgs.shape[0]

        if num_padding:
            prob = np.random.random_sample()
            if prob > 0.5:
                pad_img = imgs[0]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
            else:
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
    else:
        padded_imgs = imgs

    return padded_imgs


def predict_word_from_sign(index, tmp_img):

    transcription = {index:''}

    print('Paddding images.....')
    tmp_img = pad(tmp_img)
    print(len(tmp_img))
    print('Images padded...')

    # Run through the data augmentation
    # 64 x 224 x 224 x 3
    tmp_img = test_transforms(tmp_img)
    ret_img = video_to_tensor(tmp_img)
    inputs = ret_img[np.newaxis, ...]
    # print(inputs.shape)

# ----------------* END VIDEO PROCESSING *-----------------


    i3d = InceptionI3d(400, in_channels=3)
    # print('Loading weights.....')
    # print(i3d.state_dict().keys())
    i3d.load_state_dict(torch.load('./weights/rgb_imagenet.pt'))
    # print(i3d)
    i3d.replace_logits(NUM_CLASSES)
    i3d.load_state_dict(torch.load(
        weights,
        map_location=device))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    per_frame_logits = i3d(inputs)
    # print(per_frame_logits[0].shape)
    ## 1 x num_classes
    predictions = torch.max(per_frame_logits, dim=2)[0]
    # predictions[0] --> num_classes tensor
    # lowest as the first element - highest as the last element
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    out_probs = np.sort(predictions.cpu().detach().numpy()[0])

    gloss = get_gloss(out_labels[-1])
    print("gloss predicted: ", gloss)
    # print(out_probs)

    z = 1/(1 + np.exp(-out_probs))
    # print(z)

    # if z[-1] >= 0.0000005:
    #     transcription[index] = gloss

    transcription[index] = gloss
   
    return transcription

def batch(seq):
    sub_results = []
    for x in seq:
        sub_results.append(predict_word_from_sign(x['index'], x['img']))
    return sub_results

def get_final_phrase(index, word_list):
    word_dict = {}
    for word in word_list:
        for i in word:
            word_dict.update(i)
   
    final_string = []
    for i in range(index):
        if len(final_string)==0:
            final_string.append(word_dict.get(i, ''))
        elif final_string[-1]!=word_dict.get(i, ''):
            final_string.append(word_dict.get(i, ''))

    return ' '.join(final_string)

def video_to_text(vid):

    # ----------------* VIDEO PROCESSING *-------------------
    print('Loading images.....')
    
    imgs = load_rgb_frames_from_video(VIDEO_ROOT, vid, start=0, num = 64)
    print('Images loaded')

    all_images = []
    i=0
    j=0
    tmp_img = imgs[i:i+64]
    while len(tmp_img) >= 64:
        all_images.append({'img':tmp_img, 'index':j})
        i+=10
        j+=1
        tmp_img = imgs[i:i+64]

    all_images.append({'img':tmp_img, 'index':j})

    string_predict = []


    for i in range(0, len(all_images), 2):
        result_batch = dask.delayed(batch)(all_images[i:i+2])
        string_predict.append(result_batch)

    temp_dask = dask.delayed(get_final_phrase)(j, string_predict)
    print('ok')

    final_phrase = temp_dask.compute()
    print("Original phrase predicted is:", final_phrase)
    
    return final_phrase

# def load_rgb_frames_from_live_feed(cap, resize=(256, 256)):
#     frames = []

#     while True:
#         success, img = cap.read()
        
#         if not success:
#             print("Failed to capture frame from live feed")
#             break

#         w, h, c = img.shape
#         if w < 226 or h < 226:
#             d = 226. - min(w, h)
#             sc = 1 + d / min(w, h)
#             img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

#         if w > 256 or h > 256:
#             img = cv2.resize(img, (256, 256))

#         img = (img / 255.) * 2 - 1

#         frames.append(img)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Exiting live feed capture...")
#             break

#     print("All frames loaded from live feed")
#     return np.asarray(frames, dtype=np.float32)


# def video_to_text(cap):
#     while True:
#         print('Capturing frame...')
#         imgs = load_rgb_frames_from_live_feed(cap)
#         print('Images loaded')

#         all_images = []
#         i = 0
#         j = 0
#         tmp_img = imgs[i:i+64]
#         while len(tmp_img) >= 64:
#             all_images.append({'img':tmp_img, 'index':j})
#             i += 10
#             j += 1
#             tmp_img = imgs[i:i+64]

#         all_images.append({'img':tmp_img, 'index':j})

#         string_predict = []

#         for i in range(0, len(all_images), 2):
#             result_batch = dask.delayed(batch)(all_images[i:i+2])
#             string_predict.append(result_batch)

#         temp_dask = dask.delayed(get_final_phrase)(j, string_predict)
#         print('ok')

#         final_phrase = temp_dask.compute()
#         print("The final phrase predicted is", final_phrase)

#         cv2.imshow('Live Feed', imgs[0])  # Display the first frame of the captured video

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Exiting live feed capture...")
#             break

#     cv2.destroyAllWindows()


if __name__=='__main__':
    # final_phase = video_to_text('i_am_eating_food')
    # final_phase = video_to_text('will_you_come_to_school_tomorrow')
    # final_phase = video_to_text('hello_friend_hello')
    # final_phase = video_to_text('pray_now')
    final_phase = video_to_text('pray_now_new')
    # refined_phrase = chat_with_gpt(final_phase)
    final_phrase_str = ' '.join(final_phase)  # Convert list to a single string
    message = "The person in the video is describing various things such as: " + ', '.join(final_phase)
    
    # Refine the message with Claude
    refined_message = chat_with_claude(message)
    
    print(refined_message)
    
    # cap = cv2.VideoCapture(1)  # Change the index to the appropriate camera source
    # video_to_text(cap)
    # cap.release()