
from PyBanubaSDK.recognizer import UtilityManager, Recognizer, RecognizerMode
from PyBanubaSDK.types import FrameData, PixelFormat, FullImage, Data, FeatureId
from PyBanubaSDK.effect_player import EffectPlayer, EffectPlayerConfiguration, CameraOrientation

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import os
import glob

token = 'Qk5CICG/PtcvwvN+c3ba8+2re3eCtSC62i2FozrEaHqUf5e/CnNEpI2c2HFfzPxFR7x71pmvQuz14phE1v60scOLniGTe1yIvzrHryQzxg+us4lpJZG87oFoFQDvfjRfJsaNzJc4LGvhvt/FMcFa3GOAR4ENPIIFBNYUdh46wk8hzsZTqaxueyqBNQqBuewrnMZmrjimMbWGQ61+54n0zm2kIej/YeK0CZSCiNnP83nX/EeS5IGA0zFKVdEd9Pu4QkWXCENaXCfrxyH+B1wTAOp+ScHCHtwLc9z1iSOT2udpH7yP14y/ea1OjoR6rdVa7AIYZwkBUV+rjoBuMfgOBxdsgFNVQljeFvt1yZEnDzSzfAVW9ituQqhHBvW3BgUx+ydnVeMkx2EUb7wzfBjXoDZdfg6UCZxnijoxW2ayWJObG21tgaLpsipAglVNwd9ODlzk7/knDfD1zO6uDz06L21TDcngdDep84dvYQKL0GPu62wNAJCS9smfIgHeO3c0Qy3bd3MBX04HJPdlfTFqcveSm7wtnlFu4YXlZnDtqTbu9xkDdorGHUIlyFQSK1PZz2Ab3u0BTFv1mqzmWY+7L8Vk0seRT3PSYXcT7Cu0fhSN5SYnm0VXo5Vz9hnyn7s49QIlv+epVqApe+Ey/lPwtJC7GUeWDWksvOkg1BOXKiFfDryEZnZ1msjCUebJgLmRynEmx+y9mF5G3EF1Akd8d073NsOLXpJbpzGZ9z+FMxSkbWRoRvYlbJcS1NFxviND5NkrLfAgmOcVmpnDma1wuwV0+U+PSsz2vcEwYhc6I6atX0MAOYOt7ed8hcO94lTyRyTTQ5nT8eVSC2CmNd7joaViMCorf2PrzUkXt2sqbosAsIsBPvAezfW9oc+EE4WKSBU6Tt9R6/NUMRgWm5z4faV7xBsq4gBdenuzO6jOEFcMVLeuMqa0yVLdgdk7mNDP0JB6DSYaxHVROfocBy/4AOc77mrWW99/RNumMS/Va1fsmoDgYsFwSmrJOy6tyAnXeg5x8/hMvx9AlgLgTLkQ8WgmF6gJJ/v2XUHwH9YFZAlUhvK05Ke6oPGdJ8P9uWxuydXl3h0OuTfO2P+4pv6iQxrjzbRXmszccNMdfzAjRvJl1ZFjAiI/8R4cvQS1cc3Hopum+aRdfjn+SMrkpvz/pj0txw3FfxQqTrfZSG5KWtxcDF7Cd+1zdYivqZ255ZIKAhcJCQrn5pbh+qIeLKt2MblOstgevvKvKQSkPjfLJl33ZcOEzzC7YuR1j9WL5oefo+RKbTRz6P0Shz3Sgf3l6kwX0fSIsfyeyp59+GW8JXtZw6LHxZuh27ytSQtkQzbzJ5VNZAnYUMAe6zEZ7GbWENwlZfvXYNV4YJPgJfjnCJpbu6COr9JGA/kpG6TE+gtxso7uZ0BUp+24xK4+/OEXJJEDmCibixyHZN0aZ1rOPJ757kwc3ERu6e0a6vU93WDXFTgNImn5oCWHs0ZVZgfTcMluIh4zanZsDVI/psKMTfzrhvmsTEdzF4int7Q/d7EFmmfdZKEBqgnEVzAf8Ink0UM4OiXafeqPDHGqnVrRewuYNPJvwjmWRaiQSD8SYZe1NRwqBRgLL8G6uCAJL0ABVDdQycsGUV1KO5TNakkpP0wo0VPxYUUn+E7AjGrFyNGgOcDIDgzYl/XlcRwP4O9QAdUcNA=='

UtilityManager.initialize(list(__import__('bnb-resources').__path__), token)

from PIL import Image
# img = Image.open('10.jpg').convert("RGBA")
pixel_format = PixelFormat.Rgba

vids_dir = glob.glob('/home/ubuntu/testers/*.mp4')

# Create Recognizer and setup features to recognize
r = Recognizer.create(RecognizerMode.Synchronous)
r.set_features(set([FeatureId.Frx, FeatureId.Face]))
r.set_max_faces(1)

# Use heavy nn's (for image processing only)
r.set_offline_mode(True)

def recognizer_process(r, img, iterations=3):
    data = img.tobytes()
    # Cleanup previous state
    r.clear()
    h, w, c = np.array(img).shape
    pixel_format = PixelFormat.Rgb if c == 3 else PixelFormat.Rgba
    # Use 3 iterations for more accuracy (in case of live stream processing iterations should be 1)
    for i in range(0, iterations):
        # Create FrameData from image and process
        fd = FrameData.make_from_bpc8(data, img.width, img.height, CameraOrientation.Deg_0, pixel_format, False, 0, None)
        r.process(fd)

    # Return filled FrameData
    return fd

def is_face_found(fd, index=0):
    # Check if face was detected
    result = fd.get_frx_recognition_result()
    return result.get_faces()[index].has_face()

def dump_uint_mask(mask):
    a = np.array(mask.mask).astype(np.uint8).reshape((mask.meta.height, mask.meta.width))
    return Image.fromarray(a).convert('L')

def dump_float_mask(mask):
    a = np.array(mask.mask).reshape((mask.meta.height, mask.meta.width))
    a = a * 255
    return Image.fromarray(a).convert('L')

def transform_mask(mask, img_t, img_shape):
    img_t = np.matrix(img_t).reshape((3, 3))
    msk_t = np.matrix(mask.meta.basis_transform).reshape((3, 3))

    # (mask -> common) -> (common -> image)
    t = img_t * msk_t.I

    mask = np.array(mask.mask).astype(np.uint8).reshape((mask.meta.height, mask.meta.width))
    return Image.fromarray(mask).transform(img_shape, Image.AFFINE, data=t.I.A1)

def transform_lr_mask(mask, img_t, img_shape):
    return (transform_mask(mask.left, img_t, img_shape), transform_mask(mask.right, img_t, img_shape))

for vid in tqdm(vids_dir):
    vidObj = cv2.VideoCapture(vid)
    success = 1
    frames = []
    while success:
        success, image = vidObj.read()
        if image is None:
            continue
        image_PIL = Image.fromarray(image).convert("RGBA")
        frames.append(image_PIL)

    vid_name = os.path.basename(vid).split('.')[0]
    if os.path.exists(f'/home/ubuntu/PyBanubaSDK-1.12.0/face_analysis_vids/{vid_name}_landmark.mp4'):
        continue
    h,w,_ = np.array(frames[0]).shape
    fps=25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_lm = cv2.VideoWriter(f'/home/ubuntu/PyBanubaSDK-1.12.0/face_analysis_vids/{vid_name}_landmark.mp4', fourcc, fps, (w,h))
    out_mask = cv2.VideoWriter(f'/home/ubuntu/PyBanubaSDK-1.12.0/face_analysis_vids/{vid_name}_mask.mp4', fourcc, fps, (w,h))

    # os.makedirs(f'/home/ubuntu/PyBanubaSDK-1.12.0/lmarks_neel/{vid_name}')
    lmark_arr = []

    for i, img in tqdm(enumerate(frames)):

        fd = recognizer_process(r, img, 1)
        if not is_face_found(fd):
            img_cv2 = cv2.resize(np.array(img), (w,h))
            out_lm.write(cv2.cvtColor(img_cv2, cv2.COLOR_BGRA2BGR))

            black_img_cv2 = np.zeros(img_cv2.shape)
            out_mask.write(np.uint8(black_img_cv2))
            continue
            # raise Exception("face not found!")

        if is_face_found(fd):
            frx_result = fd.get_frx_recognition_result()
            face = frx_result.get_faces()[0]
            # face.get_landmarks() return flatten array, reshape it to get x,y points
            landmarks = np.array(face.get_landmarks()).reshape(-1, 2)

            # Draw landmarks
            landmarks_img = np.array(img, dtype='uint8')
            landmarks = np.clip(landmarks, (0, 0), (img.width, img.height)).astype(np.int32)
            for x, y in landmarks:
                landmarks_img[y:y+5, x:x+5] = [255, 0, 0, 255]
                landmarks_img[y:y-5, x:x-5] = [255, 0, 0, 255]

            # Image.fromarray(landmarks_img).save("landmarks.png")
            # np.save(f'/home/ubuntu/PyBanubaSDK-1.12.0/lmarks_neel/{vid_name}/{i}.npy', landmarks)
            lmark_arr.append(landmarks)
            landmarks_img = cv2.resize((landmarks_img), (w,h))
            out_lm.write(cv2.cvtColor(landmarks_img, cv2.COLOR_BGRA2BGR))

            # Image transformation to restore segmentation mask
            img_t = fd.get_full_img_transform()
            img_shape = (img.width, img.height)

            face_mask = transform_mask(fd.get_face(), img_t, img_shape)
            # face_mask.save('face_mask.png')
            face_mask = cv2.resize(np.array(face_mask), (w,h))
            out_mask.write(cv2.cvtColor(face_mask, cv2.COLOR_BGRA2BGR))

    np.save(f'/home/ubuntu/PyBanubaSDK-1.12.0/lmarks_neel/{vid_name}.npy', np.array(lmark_arr))
    out_lm.release()
    out_mask.release()