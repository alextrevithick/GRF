import tensorflow as tf
import numpy as np
import imageio
import random
import time
from run_nerf_helpers import *
import math

def get_similar_k(pose, pose_set, k=25):
    vp = pose[:,3]
    vp_set = pose_set[:,:,3]
    vp_set_norm = tf.norm(vp_set, axis = -1)[...,None]
    vp_norm = tf.norm(vp, axis = -1)
    simil = tf.reduce_sum( (vp / vp_norm) * (vp_set / vp_set_norm) , -1)
    return tf.argsort(simil, direction = 'DESCENDING')[:k]


transf = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1.],
        ], dtype=np.float32)


def load_pose(filename):
    assert os.path.isfile(filename)
    with open(filename) as f:
        nums = f.read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

def read_scene(scene_path, num_per_scene = -1, seed = None):
    # Get and sort files by name
    imgs = []
    poses = []
    _, _, rgb_files = next(os.walk(os.path.join(scene_path, 'rgb')))
    _, _, pose_files = next(os.walk(os.path.join(scene_path, 'pose')))
    rgb_files = sorted(rgb_files)
    pose_files = sorted(pose_files)

    num_files = len(rgb_files)

    assert num_files >= num_per_scene, scene_path
    assert len(rgb_files) == len(pose_files), scene_path

    if num_per_scene == -1:
        num_per_scene = num_files

    # Permutation of files inside of scene_path
    perm = np.random.RandomState(seed = seed).permutation(num_files)[:num_per_scene]

    im_num = 0

    for i in perm:
        rgb_file = os.path.join(scene_path, 'rgb', rgb_files[i])
        pose_file = os.path.join(scene_path, 'pose', pose_files[i])

        im = imageio.imread(rgb_file).astype(np.float32) / 255.
        alpha = np.expand_dims(im[..., 3], 2)
        im = im[..., :3] * alpha + (1. - alpha)

        pose = load_pose(pose_file)

        imgs.append(im)
        poses.append( pose @ transf )



    imgs = np.stack(imgs, 0)
    poses = np.stack(poses, 0)

    return imgs, poses



def test(render_func, scenes, args, start, one_two_recon = False, training_recon = True, render_per_scene = -1):

    datadir = args.datadir

    print("netchunk: ", args.netchunk)

    testsavedir = os.path.join(args.basedir, args.expname, 'renderonly_{}_{:06d}'.format(
        'test', start))
    os.makedirs(testsavedir, exist_ok=True)

    assert one_two_recon or training_recon, "Reconstruction from test or training must be true"

    recon_type = "train" if args.training_recon else "one_two"


    tested_scenes_file = os.path.join(args.basedir, args.expname, 'tested_scenes_{}_{}_{}.txt'.format(recon_type, str(args.from_scene), str(args.to_scene)))

    if os.path.exists(tested_scenes_file):
        print("reloading results from ", tested_scenes_file)
        with open(tested_scenes_file, "r") as f:
            tested_scenes = f.readlines()
        tested_scenes = [ line.split(" ") for line in tested_scenes if not line.isspace()]
        tested_scenes = [ (scene_info[0], scene_info[1:]) for scene_info in tested_scenes ]
        tested_scenes = dict(tested_scenes)

    else:
        with open(tested_scenes_file, "w+") as f:
            pass
        tested_scenes = {}

    if args.training_recon:
        all_psnr = [float(tested_scenes[scene_idx][0].strip()) for scene_idx in tested_scenes]
        all_ssim = [float(tested_scenes[scene_idx][1].strip()) for scene_idx in tested_scenes]
        print("current results {} {}".format(all_psnr, all_ssim))
    else:
        all_psnr_one = [float(tested_scenes[scene_idx][0].strip()) for scene_idx in tested_scenes]
        all_ssim_one = [float(tested_scenes[scene_idx][1].strip()) for scene_idx in tested_scenes]
        all_psnr_two = [float(tested_scenes[scene_idx][2].strip()) for scene_idx in tested_scenes]
        all_ssim_two = [float(tested_scenes[scene_idx][3].strip()) for scene_idx in tested_scenes]
        print("current results {} {}".format(all_psnr_one, all_ssim_one, all_psnr_two, all_ssim_two))






    to_scene = args.to_scene if args.to_scene != -1 else len(scenes)
    print("going from {} to {}".format(args.from_scene, args.to_scene))
    scenes = sorted(scenes)[args.from_scene:to_scene]

    for scene in scenes:
        init_time = time.perf_counter()
        scene_idx = scene.split("/")[-1]
        if scene_idx in tested_scenes:
            print("continuing")
            continue
        # Reconstruct train_test views from the training images
        if training_recon:
            scene_psnr = []
            scene_ssim = []

            train_test_scene = scene.split("/")
            print(train_test_scene[-1])
            train_test_scene[-2] = "train_test"
            train_test_scene = "/".join(train_test_scene)

            train_imgs, train_poses = read_scene(scene)
            train_imgs = np.stack(train_imgs, 0)
            train_poses = np.stack(train_poses, 0)
            train_poses = train_poses[:,:3]

            gt_imgs, gt_poses = read_scene(train_test_scene, num_per_scene = -1)
            gt_poses = gt_poses[:,:3]

            num_gt = len(gt_imgs)

            scene_psnr = []
            scene_ssim = []

            for gt_idx in range(num_gt):

                input_indices = get_similar_k(gt_poses[gt_idx], train_poses, k = 5)
                input_ims = tf.gather(train_imgs, input_indices)
                input_poses = tf.gather(train_poses, input_indices)

                rgb, _, _, _ = render_func(
                    c2w = gt_poses[gt_idx], attention_images = input_ims,
                    attention_poses = input_poses, render_pose=gt_poses[gt_idx])

                mse = img2mse(rgb, gt_imgs[gt_idx])
                psnr = mse2psnr(mse)

                scene_psnr.append(psnr)

                ssim = tf.image.ssim(tf.convert_to_tensor(rgb),tf.convert_to_tensor(gt_imgs[gt_idx]),max_val=1.0).numpy()
                scene_ssim.append(ssim)

                #imageio.imwrite(os.path.join(testsavedir, '{:06d}{}.png'.format(gt_idx,scene_idx)), to8b(rgb))
                #imageio.imwrite(os.path.join(testsavedir, '{:06d}{}_target.png'.format(gt_idx,scene_idx)), to8b(gt_imgs[gt_idx]))
            scene_psnr = np.mean(scene_psnr)
            scene_ssim = np.mean(scene_ssim)
            tested_scenes[scene_idx] = [scene_psnr, scene_ssim]
            with open(tested_scenes_file, 'a') as f:
                f.write("{} {} {}\n".format(scene_idx, scene_psnr, scene_ssim))
            all_psnr.append(scene_psnr)
            all_ssim.append(scene_ssim)
            print("Average time per rendering for scene", (time.perf_counter() - init_time) /  num_gt)
            print( "mean psnr: ", np.mean(all_psnr) )
            print( "mean ssim: ", np.mean(all_ssim) )



        else:
            scene_psnr_one = []
            scene_ssim_one = []
            scene_psnr_two = []
            scene_ssim_two = []

            gt_imgs, gt_poses = read_scene(scene, num_per_scene = 10)
            gt_poses = gt_poses[:,:3]

            num_gt = len(gt_imgs)


            one_recon_idx = np.random.choice(num_gt, 1)

            for gt_idx in range(num_gt):
                init_time = time.perf_counter()

                rgb, _, _, _ = render_func(
                    c2w = gt_poses[gt_idx], attention_images = gt_imgs[one_recon_idx,None],
                    attention_poses = gt_poses[one_recon_idx,None], render_pose=gt_poses[gt_idx])

                mse_one = img2mse(rgb, gt_imgs[gt_idx])
                psnr_one = mse2psnr(mse_one)

                ssim_one = tf.image.ssim(tf.convert_to_tensor(rgb),tf.convert_to_tensor(gt_imgs[gt_idx]),max_val=1.0).numpy()

                scene_psnr_one.append(psnr_one)
                scene_ssim_one.append(ssim_one)


                #imageio.imwrite(os.path.join(testsavedir, '{:06d}{}.png'.format(gt_idx,scene_idx)), to8b(rgb))
                #imageio.imwrite(os.path.join(testsavedir, '{:06d}{}_target.png'.format(gt_idx,scene_idx)), to8b(gt_imgs[gt_idx]))



            two_recon_idx = np.random.choice(num_gt, 2)

            for gt_idx in range(num_gt):
                init_time = time.perf_counter()
                rgb, _, _, _ = render_func(
                    c2w = gt_poses[gt_idx], attention_images = gt_imgs[two_recon_idx],
                    attention_poses = gt_poses[two_recon_idx], render_pose=gt_poses[gt_idx])

                mse_two = img2mse(rgb, gt_imgs[gt_idx])
                psnr_two = mse2psnr(mse_two)

                ssim_two = tf.image.ssim(tf.convert_to_tensor(rgb),tf.convert_to_tensor(gt_imgs[gt_idx]),max_val=1.0).numpy()

                all_psnr_two.append(psnr_two)
                all_ssim_two.append(ssim_two)

            scene_psnr_one = np.mean(scene_psnr_one)
            scene_ssim_one = np.mean(scene_ssim_one)
            scene_psnr_two = np.mean(scene_psnr_two)
            scene_ssim_two = np.mean(scene_ssim_two)

            tested_scenes[scene_idx] = [scene_psnr_one, scene_ssim_one, scene_psnr_two, scene_ssim_two]

            with open(tested_scenes_file, 'a') as f:
                f.write("{} {} {} {} {}\n".format(scene_idx, scene_psnr_one, scene_ssim_one, scene_psnr_two, scene_ssim_two))
            all_psnr_one.append(scene_psnr_one)
            all_ssim_one.append(scene_ssim_one)
            all_psnr_two.append(scene_psnr_two)
            all_ssim_two.append(scene_ssim_two)
            print("Average time per rendering for scene", (time.perf_counter() - init_time) /  num_gt)
            print( "mean psnr for one: ", np.mean(all_psnr_one) )
            print( "mean ssim for one: ", np.mean(all_ssim_one) )
            print( "mean psnr for two: ", np.mean(all_psnr_two) )
            print( "mean ssim for two: ", np.mean(all_ssim_two) )


            #imageio.imwrite(os.path.join(testsavedir, '{:06d}{}.png'.format(gt_idx,scene_idx)), to8b(rgb))
            #imageio.imwrite(os.path.join(testsavedir, '{:06d}{}_target.png'.format(gt_idx,scene_idx)), to8b(gt_imgs[gt_idx]))


def test_llff_blend(render_func, train_imgs, train_poses, test_imgs, test_poses, args):
    imgdir = os.path.join(args.basedir, args.expname, "render_only")
    os.makedirs(imgdir, exist_ok = True)
    print(imgdir)
    num_test = test_imgs.shape[0]
    for i in range(num_test):
        target = test_imgs[i]
        close_indices = get_similar_k(test_poses[i], train_poses, k=2)
        attention_images = tf.gather(train_imgs, close_indices)
        attention_poses = tf.gather(train_poses, close_indices)
        print(train_imgs.shape)
        print(test_imgs.shape)
        print(attention_images.shape)
        rgb,_,_,_ = render_func(c2w = test_poses[i], render_pose = test_poses[i], attention_images = attention_images, attention_poses = attention_poses)
        mse = img2mse(target, rgb)
        psnr = mse2psnr(mse)
        ssim = tf.image.ssim(tf.convert_to_tensor(rgb),tf.convert_to_tensor(test_imgs[i]),max_val=1.0).numpy()
        print(mse, psnr, ssim)
        imageio.imwrite(os.path.join(imgdir,str(i)+".png"),to8b(rgb))
        imageio.imwrite(os.path.join(imgdir,str(i)+"target.png"),to8b(target))
