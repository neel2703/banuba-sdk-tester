
from PyBanubaSDK.recognizer import UtilityManager, Recognizer, RecognizerMode
from PyBanubaSDK.types import FrameData, PixelFormat, FullImage, Data, FeatureId
from PyBanubaSDK.effect_player import EffectPlayer, EffectPlayerConfiguration, CameraOrientation

from pathlib import Path
import numpy as np

token = 'PLACE YOUR TOKEN HERE'

UtilityManager.initialize(list(__import__('bnb-resources').__path__), token)

from PIL import Image
img = Image.open('face720x1280.png').convert("RGBA")
pixel_format = PixelFormat.Rgba

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

fd = recognizer_process(r, img)
if not is_face_found(fd):
    raise Exception("face not found!")

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

Image.fromarray(landmarks_img).save("landmarks.png")


# Image transformation to restore segmentation mask
img_t = fd.get_full_img_transform()
img_shape = (img.width, img.height)

face_mask = transform_mask(fd.get_face(), img_t, img_shape)
face_mask.save('face_mask.png')
