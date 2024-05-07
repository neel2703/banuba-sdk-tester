import numpy as np
import glob
import os

lmarks = glob.glob('/home/ubuntu/PyBanubaSDK-1.12.0/lmarks_banuba/*.npy')

mismatch = []
for lmark in lmarks:
    lmark_id = os.path.basename(lmark).split('.')[0]
    banuba_lmark = np.load(lmark, allow_pickle=True)
    neel_lmark = np.load(f'/home/ubuntu/PyBanubaSDK-1.12.0/lmarks_neel/{lmark_id}.npy', allow_pickle=True)
    if neel_lmark.shape != banuba_lmark.shape:
        mismatch.append([lmark_id, str(neel_lmark.shape), str(banuba_lmark.shape)])

print(mismatch)
print(len(mismatch))