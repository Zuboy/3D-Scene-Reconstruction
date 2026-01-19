# Step 2: Common Camera Reference Frame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_world_to_camera_poses(camera_movement):
    # camera_movement[i] transforms camera i -> camera i+1.
    # Returns world->camera transforms with camera 0 at world origin.
    num_cameras = camera_movement.shape[0] + 1
    T_world_to_cam = [np.eye(4)]

    for i in range(1, num_cameras):
        # Correct chaining order: T_w2c(i+1) = T_i_to_i+1 @ T_w2c(i)
        T_world_to_cam.append(camera_movement[i - 1] @ T_world_to_cam[i - 1])

    return T_world_to_cam


def get_camera_centers(T_world_to_cam):
    centers = []
    for T in T_world_to_cam:
        R = T[:3, :3]
        t = T[:3, 3]
        centers.append(-R.T @ t)
    return np.array(centers)


T_world_to_cam = compute_world_to_camera_poses(camera_movement)
camera_centers = get_camera_centers(T_world_to_cam)

print(f"Computed {len(T_world_to_cam)} world-to-camera poses with correct order.")

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

axis_scale = 0.2

for i, T in enumerate(T_world_to_cam):
    C = camera_centers[i]
    R = T[:3, :3]

    # Camera viewing direction in world coordinates
    view_dir = R.T @ np.array([0.0, 0.0, 1.0])

    ax.scatter(C[0], C[1], C[2], color="black", s=50)
    ax.text(C[0], C[1], C[2], f" C{i}", size=10)
    ax.quiver(
        C[0],
        C[1],
        C[2],
        view_dir[0],
        view_dir[1],
        view_dir[2],
        length=axis_scale,
        color="b",
        alpha=0.8,
    )

ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], "k--", alpha=0.4)

ax.set_xlabel("X World")
ax.set_ylabel("Y World")
ax.set_zlabel("Z World")
ax.set_title("Camera Centers in Shared World Reference Frame")

max_range = np.array(
    [
        camera_centers[:, 0].max() - camera_centers[:, 0].min(),
        camera_centers[:, 1].max() - camera_centers[:, 1].min(),
        camera_centers[:, 2].max() - camera_centers[:, 2].min(),
    ]
).max() / 2.0
mid_x = (camera_centers[:, 0].max() + camera_centers[:, 0].min()) * 0.5
mid_y = (camera_centers[:, 1].max() + camera_centers[:, 1].min()) * 0.5
mid_z = (camera_centers[:, 2].max() + camera_centers[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()


# Step 3: Feature Matching and Fixed Depth Scaling
def load_depth_only(depth_path):
    files = sorted([f for f in os.listdir(depth_path) if f.lower().endswith(".png")])
    depth_images = []
    for f in files:
        # Depth images are usually 16-bit or 8-bit; IMREAD_UNCHANGED preserves this
        d = cv.imread(os.path.join(depth_path, f), cv.IMREAD_UNCHANGED)
        depth_images.append(d)
    return depth_images


DEPTH_IMGS = load_depth_only(depth_path)
print(f"Successfully loaded {len(DEPTH_IMGS)} depth maps.")

ROOM_DEPTH_M = 3.0
final_scales = [ROOM_DEPTH_M] * len(DEPTH_IMGS)

print("Using fixed depth scale for all images:")
print(f"  Z_max = {ROOM_DEPTH_M:.1f} m")
print(f"{'Images':<8} | {'Scale Value'}")
for i, s in enumerate(final_scales):
    print(f"Img {i:02d}   | {float(s):.3f}")


def get_F_matrix(K, R, t):
    t_skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = t_skew @ R
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    return F


K_mat = get_K(camera_calibration)
sorted_matches = []

for i in range(len(DEPTH_IMGS) - 1):
    T = camera_movement[i]
    R, t = T[:3, :3], T[:3, 3]
    F = get_F_matrix(K_mat, R, t)

    match_map = []
    for feat_a in given_features[i]:
        p1 = np.array([feat_a[1], feat_a[0], 1.0])
        line_in_b = F @ p1

        dists = []
        for feat_b in given_features[i + 1]:
            p2 = np.array([feat_b[1], feat_b[0], 1.0])
            d = abs(np.dot(p2, line_in_b)) / np.sqrt(line_in_b[0] ** 2 + line_in_b[1] ** 2)
            dists.append(d)
        match_map.append(np.argmin(dists))
    sorted_matches.append(match_map)

print("Epipolar matching complete for all 7 features.")

current_idx = 0
tracking = [current_idx]
for i in range(len(sorted_matches)):
    current_idx = sorted_matches[i][current_idx]
    tracking.append(current_idx)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

step3_out_path = os.path.join(out_path, "step 3 output")
os.makedirs(step3_out_path, exist_ok=True)
for i in range(len(UNDISTORTED_RGB_IMGS)):
    img = UNDISTORTED_RGB_IMGS[i].copy()
    img_draw = UNDISTORTED_RGB_IMGS[i].copy()
    img_bgr = cv.cvtColor(img_draw, cv.COLOR_RGB2BGR)
    feat_idx = tracking[i]
    h, w = given_features[i][feat_idx]

    cv.circle(img, (int(w), int(h)), 15, (0, 255, 0), -1)
    save_file = os.path.join(step3_out_path, f"tracked_view_{i:02d}.png")
    cv.imwrite(save_file, img_bgr)
    axes[i].imshow(img)
    axes[i].axis("off")

plt.tight_layout()
plt.show()


# Step 4: Depth Map to 3D Points
all_points = []
all_colors = []

fx, fy = K_mat[0, 0], K_mat[1, 1]
cx, cy = K_mat[0, 2], K_mat[1, 2]

for i in range(len(DEPTH_IMGS)):
    depth_map = DEPTH_IMGS[i]
    rgb_img = UNDISTORTED_RGB_IMGS[i]
    scale = final_scales[i]

    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_f, v_f = u.flatten(), v.flatten()

    z_f = (depth_map.flatten().astype(float) / 255.0) * scale

    mask = z_f > 0
    u_f, v_f, z_f = u_f[mask], v_f[mask], z_f[mask]

    x_c = (u_f - cx) * z_f / fx
    y_c = (v_f - cy) * z_f / fy
    z_c = z_f

    pts_hom = np.vstack((x_c, y_c, z_c, np.ones_like(z_f)))
    T_cam_to_world = np.linalg.inv(T_world_to_cam[i])
    pts_world = (T_cam_to_world @ pts_hom).T

    all_points.append(pts_world[:, :3])
    all_colors.append(rgb_img[v_f, u_f] / 255.0)

combined_pts = np.vstack(all_points)
combined_cols = np.vstack(all_colors)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

sample_idx = np.random.choice(len(combined_pts), 50000, replace=False)
pts_plot = combined_pts[sample_idx]
cols_plot = combined_cols[sample_idx]

ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], c=cols_plot, s=0.2)

for i, C in enumerate(camera_centers):
    ax.scatter(C[0], C[1], C[2], c="red", s=50)
    ax.text(C[0], C[1], C[2], f"C{i}", color="black")

ax.set_title("3D Room Reconstruction (Overlapping Views)")
ax.set_xlabel("X (World)")
ax.set_ylabel("Y (World)")
ax.set_zlabel("Z (World)")

ax.view_init(elev=-60, azim=-90)
plt.show()
