import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Alternative script that uses OpenCV rather than machinevisiontoolbox

# Load and click your points
img = cv2.imread('./images/calib_0.png', cv2.IMREAD_GRAYSCALE)
pts = []
def onclick(event):
    if event.button == 1 and len(pts) < 8:
        pts.append((event.xdata, event.ydata))
        print(f"Point {len(pts)}: {pts[-1]}")

fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Click 8 rig corners (left‑click); close when done")
plt.show()
fig.canvas.mpl_disconnect(cid)

if len(pts) != 8:
    raise RuntimeError(f"Expected 8 points, got {len(pts)}")

image_points = np.array(pts, dtype=np.float32)

# We assume the rig is set up in default configuration as shown in README
cm = 0.01
object_points = np.array([
    [ 0,  -12.2, 12.2],
    [ 0,   -6.2, 12.2],
    [ 0,  -12.2,  6.2],
    [ 0,   -6.2,  6.2],
    [ 6.2,   0,  12.2],
    [12.2,   0,  12.2],
    [ 6.2,   0,   6.2],
    [12.2,   0,   6.2],
], dtype=np.float32) * cm

obj_pts_list = [object_points]
img_pts_list = [image_points]

# FYI the flags below assume no tangential or radial distortion
# the initial guess of the intrinsic matrix is based on default camera resolution (px)
flags = (
    cv2.CALIB_ZERO_TANGENT_DIST
  | cv2.CALIB_FIX_K1               
  | cv2.CALIB_FIX_K2
  | cv2.CALIB_FIX_K3
  | cv2.CALIB_FIX_K4
  | cv2.CALIB_FIX_K5
  | cv2.CALIB_FIX_K6
  | cv2.CALIB_USE_INTRINSIC_GUESS
)

h, w = img.shape
init_K = np.array([
    [w, 0, w/2],
    [0, h, h/2],
    [0, 0,   1]
], dtype=float)

# CV2 calibration function
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts_list,
    img_pts_list,
    (w, h),
    init_K,
    None,
    flags=flags
)
if not ret:
    raise RuntimeError("Calibration failed :(")

# Explicitly zero any skew term
K[0,1] = 0

print("Intrinsic matrix K:\n", K)
print("Distortion coeffs:\n", dist.ravel())

# Save intrinsics
os.makedirs('param', exist_ok=True)
np.savetxt('param/intrinsic.txt', K, delimiter=',')

# Print extrinsics
R, _ = cv2.Rodrigues(rvecs[0])
t = tvecs[0]
print("Rotation R:\n", R)
print("Translation t:\n", t.ravel())
