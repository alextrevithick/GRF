import os
import tensorflow as tf
import numpy as np
import imageio
from functools import reduce
import itertools

def read_scene(scene_path, num_per_scene, seed, imgs, poses, im_counter):
    # Get and sort files by name
    _, _, rgb_files = next(os.walk(os.path.join(scene_path, 'rgb')))
    _, _, pose_files = next(os.walk(os.path.join(scene_path, 'pose')))
    rgb_files = sorted(rgb_files)
    pose_files = sorted(pose_files)

    num_files = len(rgb_files)
    assert num_files >= num_per_scene, scene_path
    assert len(rgb_files) == len(pose_files), scene_path

    # Permutation of files inside of scene_path
    perm = np.random.RandomState(seed = seed).permutation(num_files)[:num_per_scene]

    for i in perm:
        rgb_file = os.path.join(scene_path, 'rgb', rgb_files[i])
        pose_file = os.path.join(scene_path, 'pose', pose_files[i])

        im = imageio.imread(rgb_file)

        pose = load_pose(pose_file)

        im_num = next(im_counter)

        imgs[im_num] = im
        poses[im_num] = pose @ transf

def load_intrinsic(filename):
    with open(filename) as f:
        nums = f.read().split()
    nums = list(map(lambda x:float(x), nums))
    intrinsic = np.zeros((3,3))
    H,W = nums[-2],nums[-1]
    intrinsic[0,0] = nums[0]
    intrinsic[1,1] = nums[0] * H/W
    intrinsic[:2,2] = nums[1:3]
    intrinsic[2,2] = 1
    return intrinsic


def load_pose(filename):
    assert os.path.isfile(filename)
    with open(filename) as f:
        nums = f.read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

#invert extrinsic matrix to get pose
def extrinsic2pose(extrinsic):
    rot = extrinsic[:3,:3]
    trans = extrinsic[:3,3]
    inverse = np.zeros((4,4))
    inverse[:3,:3] = rot.T
    inverse[:3,3] = -1 * rot.T @ trans
    inverse[3,3] = 1
    return inverse

transf = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1.],
        ])

def load_shapenet_data(basedir, num_train_per_scene = 20, num_val_per_scene = 3,  num_test_per_scene = None, use_train = True, is_test = False):

    scene_dir = os.path.join(basedir, "train")

    im_counter = itertools.count()
    #random number for permutations of files
    seed_counter = itertools.count(56)

    scenes = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f))]
    _, _, rand_rgb_files = next(os.walk(os.path.join(scenes[0], 'rgb')))

    if num_train_per_scene==-1:
        num_train_per_scene = len(rand_rgb_files)
    if num_val_per_scene==-1:
        num_val_per_scene=3

    views_per_scene = num_train_per_scene + num_val_per_scene


    H, W, channels = imageio.imread(os.path.join(scenes[0], 'rgb', rand_rgb_files[0])).shape[:3]
    imgs = np.empty((views_per_scene * len(scenes), H, W, channels), dtype=np.uint8)
    poses = np.empty((views_per_scene * len(scenes), 4,4), dtype=np.float32)

    for train_scene in scenes:
        seed = next(seed_counter)
        read_scene(train_scene, num_train_per_scene, seed, imgs, poses, im_counter)
        seed = next(seed_counter)

        val_scene = train_scene.split('/')
        val_scene[-2] = 'train_val'
        val_scene = '/'.join(val_scene)
        read_scene(val_scene, num_val_per_scene, seed, imgs, poses, im_counter)


    #first num_test_scene numbers after every views_per_scene up to len(scenes)
    #thus, if num_test_scene=2, there are 5 scenes, and num_views_per_scene is 30,
    #then i_test is [[0,1],[30,31],[60,61], [90,91], [120,121]]
    i_train = [list(map(lambda x: x + y, range(num_train_per_scene))) for y in [x * views_per_scene for x in range(len(scenes))]]
    i_train = reduce(lambda x, y: x + y, i_train)
    i_val = [list(map(lambda x: x + y + num_train_per_scene, range(num_val_per_scene))) for y in [x * views_per_scene for x in range(len(scenes))]]
    i_val = reduce(lambda x, y: x + y, i_val)
    i_test = None

    poses = poses[:, :3, :4]

    render_poses = None

    i_split = [i_train, i_val, i_test]

    intrinsic = load_intrinsic(os.path.join(scenes[0], 'intrinsics.txt'))
    focal = intrinsic[0,0]
    print(imgs.shape)
    print(poses.shape)

    return imgs,poses,render_poses, [H, W, focal], i_split, [num_train_per_scene, num_val_per_scene, num_test_per_scene], intrinsic


if __name__ == '__main__':
    load_shapenet_data("/Users/alextrevithick/Desktop/synthetic_scenes", num_train_per_scene =-1)
