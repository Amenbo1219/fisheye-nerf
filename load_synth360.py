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
    with open(os.path.join(basedir, 'poses.txt')) as f:
        for line in f.readlines():
            line = line.rstrip()
            line = line.split(" ")
            print(line[0]+".png", len(line))
            pose = transform_pose(line)
            poses.append(pose)
            img = imread(os.path.join(imgdir, line[0]+".png"))/255.
            # img = imread(os.path.join(imgdir, line[0]))/255.
            img = cv2.resize(img, (1024, 1024))
            # img = cv2.resize(img, (320, 160))
            imgs.append(img)

    imgs = np.array(imgs).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    return poses, imgs
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
    train = basedir+'/train/'
    test = basedir+'/test/'
    t_poses, t_images = _load_data(train)
    l_poses, l_images = _load_data(test)
    images = np.concatenate([t_images,l_images],0)
    poses = np.concatenate([t_poses,l_poses],0)
    poses = normalize(poses)
    print(poses)
    bds = np.array([images.shape[1], images.shape[2], None])

    # NeRF座標系に逆変換
    # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # imgs = np.moveaxis(images, -1, 0).astype(np.float32)
    # images = imgs
    
    # bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # print(images.shape,poses.shape)
    i_test = np.array([i for i in range(t_images.shape[0],l_images.shape[0]+t_images.shape[0])])
    # i_test = l_images.shape
    # この辺は確認用のRendringPathなので過度に気にしなくてOK
    # c2w_path = poses_avg(poses)
    # up = normalize(poses[:, :3, 1].sum(0))
    # tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    # rads = np.percentile(np.abs(tt), 90, 0)
    # dt = .75
    # # close_depth, inf_depth = np.ravel(bds).min()*.9, np.ravel(bds).max()*5.
    # # mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    # focal = 1
    # zdelta =  images.shape[1]*.9 * .2
    # N_rots = 2
    # N_views = 120
    # render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses=interpolate_poses(poses).astype(np.float32)
    # render_poses = np.array(poses).astype(np.float32)
    # # 
    # render_poses = np.array(l_poses).astype(np.float32)
    return images, poses, bds ,render_poses,i_test