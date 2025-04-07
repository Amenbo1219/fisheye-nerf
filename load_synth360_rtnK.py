import os
import numpy as np
import cv2
import torch 
import math
def roll_rotation_matrix(angle_degrees):
    angle_radians = math.radians(angle_degrees)
    return np.array([
        [1, 0, 0],
        [0, math.cos(angle_radians), -math.sin(angle_radians)],
        [0, math.sin(angle_radians), math.cos(angle_radians)]
    ])
def detect_fisheye_crop_and_get_coords_with_mask(img, out_size=1024, fname=None, basedir=None,scale_factor=1/4):
    original_H, original_W = img.shape[:2]

    # 1/8サイズで高速処理
    resized_img = cv2.resize(img, (int(original_W * scale_factor), int(original_H * scale_factor)))
    gray = cv2.cvtColor((resized_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # マスク用空配列（元サイズ）
    mask = np.zeros((original_H, original_W), dtype=np.uint8)

    # 円検出
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=10, maxRadius=gray.shape[0] // 2
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 1つ目の円を取得してスケーリング
        x, y, r = map(int, circles[0][0])
        x = int(x / scale_factor)
        y = int(y / scale_factor)
        r = int(r / scale_factor)
        box_size = int(r * 2)

        # マスク生成（スケーリング後の座標で元サイズに）
        cv2.circle(mask, (x, y), r, 255, -1)

        # マスク保存
        if fname is not None and basedir is not None:
            outname = "mask_L.png" if "_L" in fname else "mask_R.png" if "_R" in fname else "mask.png"
            outpath = os.path.join(basedir, outname)
            if not os.path.exists(outpath):
                cv2.imwrite(outpath, mask)

        # クロップ領域
        left = max(x - r, 0)
        top = max(y - r, 0)
        right = min(x + r, original_W)
        bottom = min(y + r, original_H)

        cropped = img[top:bottom, left:right]

        # 黒キャンバスに中心配置
        canvas = np.zeros((box_size, box_size, 3), dtype=np.float32)
        ch, cw = cropped.shape[:2]
        start_y = (box_size - ch) // 2
        start_x = (box_size - cw) // 2
        canvas[start_y:start_y+ch, start_x:start_x+cw] = cropped

        final = cv2.resize(canvas, (out_size, out_size))
        crop_info = (top, bottom, left, right, box_size)
        return final, crop_info, mask

    else:
        print(f"⚠️ 円検出失敗: {fname}")
        resized = cv2.resize(img, (out_size, out_size))
        crop_info = (0, original_H, 0, original_W, original_H)
        empty_mask = np.zeros((original_H, original_W), dtype=np.uint8)
        return resized, crop_info, empty_mask
def transform_pose(line):
    # 元の姿勢行列を作成
    # make_transform_pose
    X_FLIP = np.diag([-1.0, 1.0, 1.0])# X軸反転
    transform_pose = np.zeros((4,4),np.float32)
    transform_pose[3,3] = 1.0
    # make_rotationMATRIX
    rotation_matrix = np.array([float(x) for x in line[1:10]]).reshape(3,3)
    translation = np.array([float(x) for x in line[10:13]]).reshape(3)
    rotation_matrix = rotation_matrix @ X_FLIP  
    translation = translation @ X_FLIP 
    # rotation_matrix = YAW_FIX@ rotation_matrix
    # roll_rotation = roll_rotation_matrix(-270)
    # -90Degree_FIXed
    # rotation_matrix = roll_rotation @ rotation_matrix
    # Applyed RotationMatrix
    transform_pose[:3,:3] = rotation_matrix
    transform_pose[0:3,3] = translation.T

    return transform_pose
def _load_data(basedir):
    imgdir = os.path.join(basedir, 'images')

    def imread(f):
        return cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)

    poses = []
    imgs = []
    masks = []
    crop_cache = {}

    with open(os.path.join(basedir, 'poses.txt')) as f:
        for line in f.readlines():
            line = line.rstrip().split(" ")
            fname = line[0] + ".png"
            pose = transform_pose(line)
            poses.append(pose)

            img = imread(os.path.join(imgdir, fname)) / 255.0

            if fname not in crop_cache:
                # 初回検出
                processed_img, crop_info, mask = detect_fisheye_crop_and_get_coords_with_mask(
                    img, fname=fname, basedir=basedir
                )
                crop_cache[fname] = (crop_info, mask)
            else:
                crop_info, mask = crop_cache[fname]
                top, bottom, left, right, box_size = crop_info
                cropped = img[top:bottom, left:right]

                canvas = np.zeros((box_size, box_size, 3), dtype=np.float32)
                ch, cw = cropped.shape[:2]
                start_y = (box_size - ch) // 2
                start_x = (box_size - cw) // 2
                canvas[start_y:start_y+ch, start_x:start_x+cw] = cropped
                processed_img = cv2.resize(canvas, (1024, 1024))

            imgs.append(processed_img)

            # マスクも1024x1024に揃える（白黒2値）
            resized_mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            masks.append(resized_mask.astype(np.uint8))

    imgs = np.array(imgs).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    masks = np.array(masks).astype(np.uint8)  # (N, 1024, 1024)

    return poses, imgs, masks


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m
def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

# import numpy as np

def interpolate_poses(poses, target_frames=200):
    """
    poses: (N, 4, 4) のPose行列（同次変換）
    return: (target_frames, 4, 4) のPose行列（位置のみ等間隔補間）
    """
    positions = poses[:, :3, 3]     # (N, 3)
    num_original = positions.shape[0]

    # 等間隔な補間インデックス（float）
    interp_indices = np.linspace(0, num_original - 1, target_frames)

    interp_positions = []
    for idx in interp_indices:
        low = int(np.floor(idx))
        high = min(low + 1, num_original - 1)
        t = idx - low
        interp_pos = (1 - t) * positions[low] + t * positions[high]
        interp_positions.append(interp_pos)
    interp_positions = np.array(interp_positions)  # (target_frames, 3)

    # 回転（+その他）は最初のPoseの R を流用（3x3）
    rot = poses[0, :3, :3]

    # Pose再構築
    new_poses = []
    for i in range(target_frames):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot
        pose[:3, 3] = interp_positions[i]
        new_poses.append(pose)

    return np.stack(new_poses, axis=0)  # (target_frames, 4, 4)


def normalize(poses):
    SCALE = 2.0  # 1 unit → 1m に合わせたいなら
    # normalized_matrices にスケーリングされた行列が格納されている
    poses[:,0,3]*= SCALE
    poses[:,2,3]*= SCALE
    
    return poses
# def load_synth360_data(basedir):
#     poses, images = _load_data(basedir)

#     bds = np.array([images.shape[1], images.shape[2], None])

#     return images, poses, bds
def load_synth360_data(basedir):
    train = os.path.join(basedir, 'train')
    test = os.path.join(basedir, 'test')
    t_poses, t_images,masks = _load_data(train)
    l_poses, l_images,_ = _load_data(test)

    images = np.concatenate([t_images, l_images], axis=0)
    poses = np.concatenate([t_poses, l_poses], axis=0)
    poses = normalize(poses)

    H, W = images.shape[1], images.shape[2]
    bds = np.array([W, H, None])

    # --- 焦点距離導出 (理論に基づく) ---
    sensor_width_mm = 8.7552  # mm
    f_mm = 2.57               # mm
    pixel_size_mm = sensor_width_mm / W  # (1)
    fx = fy = f_mm / pixel_size_mm       # (2)
    cx = W / 2
    cy = H / 2

    K = np.array([
        [fx, 0, cx],
        [0,  fy, cy],
        [0,   0,  1]
    ], dtype=np.float32)

    i_test = np.array([i for i in range(t_images.shape[0], l_images.shape[0] + t_images.shape[0])])
    render_poses = interpolate_poses(poses).astype(np.float32)

    return images, poses, bds, render_poses, i_test, K,masks