from realsense import MultiRealsense
import numpy as np
import time

realsenses = MultiRealsense(enable_depth=True)
realsenses.start()


print("Wating for 3 seconds")
time.sleep(3)

print("Captureing")
intrinsics = realsenses.get_intrinsics()
depth_scale = realsenses.get_depth_scale()
res = realsenses.get()
keys = list(res.keys())

cam_1 = keys[0]
K = intrinsics[cam_1]
depth_scale = depth_scale[cam_1]
depth = res[cam_1]['depth']
color = res[cam_1]['color']
print(depth[100: 200, 100:200])

save_data = {
    "K": K,
    "depth_scale": depth_scale,
    "depth": depth,
    'color': color
}

np.save("test_data.npy", save_data, allow_pickle=True)



realsenses.stop()
