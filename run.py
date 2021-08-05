import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu_num = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num  ## specify the GPU to use
config = tf.ConfigProto()
tf.compat.v1.enable_eager_execution(config=config)

import sys
import numpy as np
import imageio
import random
import time
from run_nerf_helpers import *
from load_shapenet_rewrite_lowmem import load_shapenet_data
from load_llff import load_llff_data
import math
import test_shapenet
from queue import Queue
from make_models import init_nerf_attention_model, init_nerf_model, init_unet
from matrix_to_quat import matrix2quat
from load_blender import load_blender_data


#pts [n_pts,3]: points in 3d space
#attention_poses[n_views,3,4]: matrices corresponding to the different viewpoints of a given scene
#intrinsic[3,4]: intrinsic matrix
#returns image plane pixel locations of rays originating at all of the
#attention_poses and going through one of the given points. The output tensor has shape [n_views,n_pts,2]
@tf.function
def make_indices(pts, attention_poses, intrinsic, H, W):
    hom_points = tf.concat([pts, tf.broadcast_to([1.0], pts.shape[:-1] + (1,))], -1)
    extrinsic = invert(attention_poses)[:, :3]
    focal = intrinsic[0, 0]

    pt_camera = tf.broadcast_to(hom_points[None, ...],
                                (extrinsic.shape[0], hom_points.shape[0], hom_points.shape[1])) @ tf.transpose(
        extrinsic, [0, 2, 1])
    pt_camera = focal / pt_camera[:, :, 2][..., None] * pt_camera

    final = 1.0 / focal * (pt_camera @ tf.transpose(intrinsic))
    final = tf.reverse(final, axis=[-1])[..., 1:]
    final = (tf.constant([0., W]) - final) * tf.constant([-1., 1.])
    final = tf.round(final)
    final = tf.math.maximum(tf.math.minimum(final, [H-1.,W-1.]),0)
    final = tf.cast(final, dtype=tf.int32)

    return final
@tf.function
def make_llff_indices(pts, attention_poses, intrinsic, H, W):
    hom_points = tf.concat([pts, tf.broadcast_to([1.0], pts.shape[:-1] + (1,))], -1)
    extrinsic = invert(attention_poses)[:, :3]
    focal = intrinsic[0, 0]

    pt_camera = tf.broadcast_to(hom_points[None, ...],
                                (extrinsic.shape[0], hom_points.shape[0], hom_points.shape[1])) @ tf.transpose(
        extrinsic, [0, 2, 1])
    pt_camera = focal / pt_camera[:, :, 2][..., None] * pt_camera

    final = 1.0 / focal * (pt_camera @ tf.transpose(intrinsic))
    final = tf.reverse(final, axis=[-1])[..., 1:]
    final = (tf.constant([0., W]) - final) * tf.constant([-1., 1.])
    final = tf.round(final)



    final = tf.cast(final, tf.int32)

    out1 = final > tf.constant([H-1, W-1], dtype = tf.int32)
    out1 = tf.reduce_any(out1, -1, keepdims = True)

    out2 = final < 0
    out2 = tf.reduce_any(out2, -1, keepdims = True)

    out = tf.logical_or(out1, out2)
    out_mask = tf.cast(tf.broadcast_to(out, out.shape[:-1] + (2,)), tf.int32)
    out = out_mask * tf.constant([H, W])
    in_mask = (out_mask - 1)* -1

    final = final * in_mask + out

    return final

def gather_indices(pts, attention_poses, intrinsic, images_features):
    #do the clipping here and append unclipped
    H,W = images_features.shape[1:3]
    H=int(H)
    W=int(W)
    indices = make_indices(pts, attention_poses, intrinsic, H, W)
    features = tf.gather_nd(images_features, indices, batch_dims = 1)
    return tf.concat([features, tf.cast(indices, dtype=tf.float32) ], -1)

def batchify_cache(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs, training = False):

        ret_list = [fn(inputs[i:i+chunk], training=training) for i in range(0, int(inputs.shape[0]), chunk)]

        return tf.concat([ret for ret in ret_list], 0)
    return ret


def batchify(fn, chunk, world_fn = lambda x:x, gather_func = None):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn
    def ret(inputs, training = False, world_fn=world_fn):
        embedded = inputs[0]
        attention_poses = inputs[1]
        intrinsic = inputs[2]
        images_features = inputs[3]
        pts = inputs[4]

        ret_list = [fn([embedded[i:i+chunk], gather_func( world_fn(pts[i:i+chunk]), attention_poses, intrinsic, images_features),pts[i:i+chunk] ]
        , training=training) for i in range(0, int(embedded.shape[0]), chunk)]
        #necessary to cache computed results from coarse model
        if fn.coarse:
            return tf.concat([pred[0] for pred in ret_list], 0), tf.concat([pred[1] for pred in ret_list], 0)
        else:
            return tf.concat([pred[0] for pred in ret_list], 0), None
    return ret

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                fine_cache_query,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                attention_poses=None,
                intrinsic=None,
                verbose=False,
                training=False,
                images_features=None):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        #integration time. The period for which a noisy signal is averaged in order
        # to improve the signal to noise ratio in an electronic system. See sensitivity.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ##############################

    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None #N_rays, 3

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Evaluate model at each point.
    raw, attention_cache = network_query_fn(pts, viewdirs, network_fn, attention_poses, intrinsic, training, images_features)  # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    fine_cache = fine_cache_query(attention_cache)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.concat([z_vals, z_samples], -1)
        inds = tf.argsort(z_vals, -1)
        z_vals = tf.sort(z_vals, -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_samples[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine
        raw_fine, _ = network_query_fn(pts, viewdirs, run_fn, attention_poses, intrinsic, training, images_features)



        fine_cache = tf.reshape(fine_cache, [raw_fine.shape[0],-1,raw_fine.shape[2]])

        raw = tf.gather(tf.concat([fine_cache, raw_fine], 1),inds,batch_dims=1)


        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)


    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret



def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           attention_images=None, attention_poses=None, intrinsic=None, render_pose=None,
           attention_embed_fn=None, attention_embed_ln=None, unet_model=None,rotation_embed_fn = None, rotation_embed_ln = None, use_render_pose = True,
           **kwargs):

    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)

    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)

    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)



    viewpoints = attention_poses[...,3]
    embedded_viewpoints = attention_embed_fn(viewpoints)
    bc_viewpoints = tf.broadcast_to(embedded_viewpoints[:,None,None], attention_images.shape[:-1] + (attention_embed_ln,))
    if use_render_pose:
        bc_render_transl = tf.broadcast_to(attention_embed_fn(render_pose[...,3])[None,None,None], attention_images.shape[:-1] + (attention_embed_ln,))
        bc_viewpoints = tf.concat([bc_viewpoints, bc_render_transl], -1)


    if rotation_embed_fn is not None:
        attention_quats = matrix2quat(attention_poses[:,:3,:3])
        attention_quats = rotation_embed_fn(attention_quats)
        attention_quats = tf.broadcast_to(attention_quats[:,None,None], attention_images.shape[:-1] + (rotation_embed_ln,) )
        if use_render_pose:
            render_quat = matrix2quat(render_pose[:3,:3])
            render_quat = rotation_embed_fn(render_quat)
            render_quat = tf.broadcast_to(render_quat[None,None,None], attention_images.shape[:-1] + (rotation_embed_ln,) )
            quats = tf.concat([attention_quats, render_quat], -1)
        else:
            quats = attention_quats

        rgb_vp=tf.concat([attention_embed_fn(attention_images),bc_viewpoints, quats],-1)
    else:
        rgb_vp=tf.concat([attention_embed_fn(attention_images),bc_viewpoints],-1)

    images_features = unet_model(rgb_vp)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, attention_poses = attention_poses, intrinsic=intrinsic, images_features=images_features, **kwargs)
    for k in all_ret:
        if k is not 'max_rel' and k is not 'max_rel_fine':
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]




def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*32, attention_poses=None, intrinsic=None, training = False, images_features = None, world_fn = None, gather_func = None):
    """Prepares inputs and applies network 'fn'."""
    #flattened points
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded,embedded_dirs],-1)

    outputs_flat, attention_cache = batchify(fn, netchunk,
        world_fn = world_fn, gather_func = gather_func)([embedded,
        attention_poses, intrinsic, images_features, inputs_flat], training)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs, attention_cache



#num_features is output length of attention/slot attention model to nerf model
def create_model(args, H, W, focal, num_features=256):

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]

    attention_embed_fn, attention_embed_ln = get_embedder(5,0)
    attention_embed_fn_2, attention_embed_ln_2 = get_embedder(2,0,9)

    if args.dataset_type == 'llff' or args.use_quaternion: #or args.dataset_type == 'shapenet':
        rotation_embed_fn, rotation_embed_ln = get_embedder(2,0,4)
    else:
        rotation_embed_fn, rotation_embed_ln = None, 0
    print("use quaternions: ", rotation_embed_ln != 0)
    print("use globl: ", not args.no_globl)
    print("use render pose", not args.no_render_pose)
    unet_model_obj = init_unet(attention_embed_ln,dtype = args.dataset_type, rotation_embed_ln = rotation_embed_ln, use_globl = not args.no_globl, use_render_pose = not args.no_render_pose)

    grad_vars = unet_model_obj.trainable_variables



    print("use attsets: ", args.use_attsets)
    if args.use_attsets:
        from attsets import attsets
        num_features = 512
        attention_module = attsets(attention_output_length = num_features)
    else:
        from slot_attention_module import slot_attention
        if args.dataset_type == 'shapenet':
            hidden_dim = 256
            iters = 3
        else:
            hidden_dim = 128
            iters = 2
        num_slots = 2
        num_features = num_slots * hidden_dim
        attention_module = slot_attention(num_slots, hidden_dim, attention_output_length = num_features, iters = iters)
        print("num_slots: ", attention_module.num_slots)
        print("iters: ", attention_module.iters)
    #set input shape so that the layer is built
    samp_embedded_pts = tf.ones((2,2,input_ch))
    #this is 128+2+att_len because we have 128 features in output in unet, and we append embedded_rgb and also the 2 indices
    samp_input = tf.ones((2,2,128 + 2 + attention_embed_ln))

    attention_module(samp_input, samp_embedded_pts)
    grad_vars += attention_module.trainable_variables

    nerf_model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, image_features = num_features)
    model = init_nerf_attention_model(nerf_model, attention_module, attention_embed_fn, input_ch,attention_embed_fn_2, attention_embed_ln_2, True, args.N_samples)

    grad_vars += nerf_model.trainable_variables
    models = {'model': model}
    models['attention_model'] = attention_module
    models['unet_model'] = unet_model_obj

    model_fine = None
    if args.N_importance > 0:
        nerf_model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, image_features=num_features)
        model_fine = init_nerf_attention_model(nerf_model_fine, attention_module, attention_embed_fn, input_ch,attention_embed_fn_2, attention_embed_ln_2, False, args.N_importance+args.N_samples)
        models['model_fine'] = model_fine
        grad_vars += nerf_model_fine.trainable_variables

    if args.dataset_type == 'llff':
        world_fn = lambda ndc_pts: ndc2world(H, W, float(focal), 1., ndc_pts)
    else:
        world_fn = lambda x: x
    index_func = make_llff_indices if args.dataset_type == 'llff' else make_indices
    def gather_indices(pts, attention_poses, intrinsic, images_features):
        #do the clipping here and append unclipped
        H,W = images_features.shape[1:3]
        H=int(H)
        W=int(W)
        indices = index_func(pts, attention_poses, intrinsic, H, W)
        features = tf.gather_nd(images_features, indices, batch_dims = 1)
        return tf.concat([features, tf.cast(indices, dtype=tf.float32) ], -1)

    def network_query_fn(inputs, viewdirs, network_fn, attention_poses, intrinsic, training, images_features):
        return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk, attention_poses=attention_poses,
        intrinsic=intrinsic, training=training, images_features
        = images_features, world_fn = world_fn, gather_func = gather_indices)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'training': True,
        'fine_cache_query': batchify_cache(nerf_model_fine, 4 * args.netchunk),
        'unet_model': unet_model_obj,
        'attention_embed_fn':attention_embed_fn,
        'attention_embed_ln': attention_embed_ln,
        'rotation_embed_fn':rotation_embed_fn,
        'rotation_embed_ln': rotation_embed_ln,
        'use_render_pose': not args.no_render_pose
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['training'] = False

    start = 0
    basedir = args.basedir
    expname = args.expname
    print("learning rate: ", args.lrate, "decay: ", args.lrate_decay)
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(args.lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f and 'attention' not in f and 'unet' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]

        ft_weights_optimizer = '{}optimizer_{}'.format(ft_weights[:-16], ft_weights[-10:])
        print("Reloading optimizer from", ft_weights_optimizer)
        opt_weights = np.load(ft_weights_optimizer, allow_pickle=True)
        if not len(opt_weights) == 0:
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            optimizer.apply_gradients(zip(zero_grads, grad_vars))
            optimizer.set_weights(opt_weights)
        else:
            print("Saved optimizer was empty, optimizer has been reinitialized")

        print('Reloading coarse from', ft_weights)
        model.nerf_model.set_weights(np.load(ft_weights, allow_pickle=True))

        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.nerf_model.set_weights(np.load(ft_weights_fine, allow_pickle=True))

        ft_weights_attention = '{}attention_model_{}'.format(ft_weights[:-16],ft_weights[-10:])
        print('Reloading slot attention from', ft_weights_attention)
        attention_module.set_weights(np.load(ft_weights_attention, allow_pickle=True))

        ft_weights_unet = '{}unet_model_{}'.format(ft_weights[:-16],ft_weights[-10:])
        print('Reloading unet from', ft_weights_unet)
        unet_model_obj.set_weights(np.load(ft_weights_unet, allow_pickle=True))

        models['optimizer'] = optimizer


    return render_kwargs_train, render_kwargs_test, start, grad_vars, models, optimizer

def load_data(args):
    if args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split, data_split, intrinsic = load_shapenet_data(args.datadir)

        print('Loaded deepvoxels', images.shape,
               hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

        inp_img_range = (2,3)

        num_accum = 3

        intrinsic = np.array([[525., 0., 256.], [0., 525., 256.], [0., 0, 1]])

        return images, poses, render_poses, [i_train, i_val, i_test], near, far, hwf, intrinsic, data_split, inp_img_range, num_accum
    #This is the only loading method we use, but we could use deepvoxels...
    elif args.dataset_type == 'shapenet':

        images, poses, render_poses, hwf, i_split, data_split, intrinsic = load_shapenet_data(args.datadir)

        print('Loaded shapenet', images.shape,
               hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

        inp_img_range = (2,6)

        num_accum = 3

        return images, poses, render_poses, [i_train, i_val, i_test], near, far, hwf, intrinsic, data_split, inp_img_range, num_accum

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near = 2.
        far = 6.
        H, W, focal = hwf
        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]
        train_imgs = images[i_train]
        train_poses = poses[i_train]
        val_imgs = images[i_val]
        val_poses = poses[i_val]
        test_imgs = images[i_test]
        test_poses = poses[i_test]
        i_train = range(train_imgs.shape[0])
        if args.render_only:
            i_test = [x + train_imgs.shape[0] for x in range( test_imgs.shape[0] )]
            i_val = None
            images = np.concatenate([train_imgs, test_imgs], 0)
            poses = np.concatenate([train_poses, test_poses], 0)
        else:
            i_test = None
            i_val = [x + train_imgs.shape[0] for x in range( val_imgs.shape[0] )]
            images = np.concatenate([train_imgs, val_imgs], 0)
            poses = np.concatenate([train_poses, val_poses], 0)
        intrinsic = np.array([[focal, 0., W/2],
                                [0, focal, H/2],
                                [0, 0, 1.]])
        data_split = [train_imgs.shape[0], val_imgs.shape[0], test_imgs.shape[0]]
        inp_img_range =(2,3)
        num_accum = 1
        poses = poses[:,:3,:4]
        return images, poses, render_poses, [i_train, i_val, i_test], near, far, hwf, intrinsic, data_split, inp_img_range, num_accum
    elif args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print(i_train)

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        train_imgs = images[i_train]
        train_poses = poses[i_train]

        if i_val is not None:
            val_imgs = images[i_val]
            val_poses = poses[i_val]

        if i_test is not None:
            test_imgs = images[i_test]
            test_poses = poses[i_test]

        inp_img_range = (2,3)



        H,W, focal = float(hwf[0]), float(hwf[1]), float(hwf[2])

        intrinsic = np.array([[focal, 0., W/2],
                                [0, focal, H/2],
                                [0, 0, 1.]])


        data_split = [int(train_imgs.shape[0]), int(val_imgs.shape[0]), None]

        i_train = list(range(data_split[0]))

        i_train = range(train_imgs.shape[0])
        if args.render_only:
            i_test = [x + train_imgs.shape[0] for x in range( test_imgs.shape[0] )]
            i_val = None
            images = np.concatenate([train_imgs, test_imgs], 0)
            poses = np.concatenate([train_poses, test_poses], 0)
        else:
            i_test = None
            i_val = [x + train_imgs.shape[0] for x in range( val_imgs.shape[0] )]
            images = np.concatenate([train_imgs, val_imgs], 0)
            poses = np.concatenate([train_poses, val_poses], 0)

        num_accum = 3


        return images, poses, render_poses, [i_train,i_val,i_test], near, far, hwf, intrinsic, data_split, inp_img_range, num_accum
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

def aggregate_rays(H, W, focal, poses, images):
    # For random ray batching.
    #
    # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
    # interpreted as,
    #   axis=0: ray origin in world space
    #   axis=1: ray direction in world space
    #   axis=2: observed RGB color of pixel
    print('get rays')
    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
    # for each pixel in the image. This stack() adds a new dimension.
    rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
    rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
    print('done, concats')
    # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
    # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
    #rays_rgb = np.stack([rays_rgb[i]
                         #for i in i_train], axis=0)  # train images only
    # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = np.reshape(rays_rgb, [rays_rgb.shape[0],H*W, 3, 3])
    print(rays_rgb.shape)
    rays_rgb = rays_rgb.astype(np.float32)
    return rays_rgb


def train():
    print("fc")
    parser = config_parser()
    args = parser.parse_args()

    if args.render_only and args.dataset_type=='shapenet':
        print('RENDER ONLY')

        logdir = os.path.join(args.basedir, args.expname)

        near, far, H, W = parse_attributes(logdir)

        if args.training_recon:
            scene_dir = os.path.join(args.datadir, "train")
        else: #one_two_recon must be True
            scene_dir = os.path.join(args.datadir, "test")

        scenes = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f))]

        intrinsic = load_intrinsic(os.path.join(scenes[0], "intrinsics.txt"))
        intrinsic = tf.cast(intrinsic, tf.float32)
        focal = intrinsic[0,0]

        _, render_kwargs_test, start, _, _, _ = create_model(args, H, W, focal)

        bds_dict = {
            'near': tf.cast(near, tf.float32),
            'far': tf.cast(far, tf.float32),
        }

        # Model must know the near and far estimates from the training data
        render_kwargs_test.update(bds_dict)

        render_func = lambda c2w, attention_images, attention_poses, render_pose : render(
            H, W, focal, c2w = c2w, attention_images = attention_images,
            attention_poses = attention_poses, intrinsic = intrinsic, render_pose=render_pose, **render_kwargs_test)

        print("training_recon", args.training_recon)
        print("render_per_scene", args.render_per_scene)


        test_shapenet.test(render_func, scenes, args, start, render_per_scene=args.render_per_scene)

        return

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    images, poses, render_poses, i_split, near,far, hwf, intrinsic, data_split, inp_img_range, num_accum = load_data(args)
    intrinsic = tf.cast(intrinsic, tf.float32)

    num_train_per_scene, num_val_per_scene, num_test_per_scene = data_split
    views_per_scene = num_train_per_scene + num_val_per_scene


    i_train, i_val, i_test = i_split
    num_test_scenes = max(i_train)//views_per_scene



    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]


    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models, optimizer = create_model(
        args, H, W, focal)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)



    attrib_f = os.path.join(basedir, expname, 'scene_attributes.txt')

    if not os.path.exists(attrib_f):
        with open(attrib_f, 'w') as file:
            file.write(" ".join([str(near), str(far), str(H), str(W)]))

    # Create optimizer
    lrate = args.lrate
    print("learning rate",lrate)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    print("USE_BATCHING", use_batching)

    assert not (use_batching and not args.no_render_pose), "Can't set both use batching and use render pose"


    i_batch = 0

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    acm_grads = None
    acm_time = time.time()
    acm_loss = 0.

    num_models_cp = 3
    last_saved_dict = {'optimizer' : Queue(num_models_cp), 'model' : Queue(num_models_cp), 'attention_model' :
        Queue(num_models_cp), 'model_fine' : Queue(num_models_cp), 'unet_model' : Queue(num_models_cp)}

    render_pose = None

    if use_batching:
        processed_images  = preprocess_images(images) if args.dataset_type=='shapenet' or args.dataset_type=='deepvoxels' else images
        rays_rgb = aggregate_rays(H,W,focal, poses, processed_images)
        num_accum = 1
    
    print("num_accum: ", num_accum)
    print("render_only")
    def get_data(scene_i, num_images):
        p1 = np.arange(scene_i * views_per_scene, scene_i * views_per_scene + num_train_per_scene)
        
        if args.dataset_type == 'shapenet':
            perm = np.random.permutation(num_train_per_scene)[:num_images]
            p1 = p1[perm]
            if img_i in p1:
                p1 = np.delete(p1, np.where(p1==img_i))
            attention_images = preprocess_images(images[p1])
            attention_poses = poses[p1,:3,:4]
            return attention_poses, attention_images
        elif args.dataset_type == 'llff' or args.dataset_type == 'blender':
            current_poses = np.take(poses, p1, 0)[:,:3,:4]
            current_images = np.take(images, p1, 0) #top size was set at 4 for reflective ones
            return get_similar_k(pose, current_poses, current_images, top_size = min(num_train_per_scene,20), num_from_top = num_images)
    def read_im(image):
        if args.dataset_type == 'shapenet' or args.dataset_type == 'deepvoxels':
            return preprocess_images(image)
        else:
            return image


    if args.render_only:
        all_ssim = []
        all_psnr = []
        savedir = os.path.join(args.basedir, args.expname, "paper_test_imgs")
        try:
            os.makedirs(savedir, exist_ok = True)
            print("save dir created at ", savedir)
        except:
            print("Directory not created") 
        render_list = [] 
        assert i_val is None, "no val during testing"
        views_per_scene = len(i_train)+len(i_test)
        count = -1
        scene_i =0
        img_i =0
        for img_i in i_test:
             init_time = time.time()
             target = read_im(images[img_i])
             scene_i = int(img_i / views_per_scene)
             scene_train_idxs = np.arange(scene_i * views_per_scene, scene_i * views_per_scene + num_train_per_scene)
             print(scene_train_idxs)
             print(scene_i)
             pose = poses[img_i, :3, :4]
             
             print("img_i")
             print(img_i)
             num_images = 4
             train_imgs, train_poses = images[scene_train_idxs], poses[scene_train_idxs]
     
             attention_poses, attention_images = get_similar_k(pose, train_poses, train_imgs, k=4)
             
             rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                    attention_images = attention_images,
                                                    attention_poses = attention_poses, intrinsic=intrinsic,render_pose=pose,
                                                    **render_kwargs_test)


             psnr = mse2psnr(img2mse(rgb, target)).numpy()
             ssim = tf.image.ssim(tf.convert_to_tensor(rgb),tf.convert_to_tensor(target),max_val=1.0).numpy()
             all_psnr.append(psnr)
             all_ssim.append(ssim)
             render_list.append(rgb.numpy())
             imageio.imwrite(os.path.join(savedir, '{}{}.png'.format(str(img_i),"rendert")), to8b(rgb))

             imageio.imwrite(os.path.join(savedir, '{}{}.png'.format(str(img_i),"target")), to8b(target))

             #imageio.imwrite(os.path.join(savedir, '{}{}.png'.format(str(img_i),"input")), to8b(attention_images[0]))
             #print(expname, psnr, ssim, global_step.numpy())
             print( (time.time() - init_time)/60.)
        imageio.mimwrite(os.path.join(savedir, 'final_vid.mp4'), to8b(render_list), fps=30, quality=8)
        print("results")
        print("psnr: ", np.mean(all_psnr))
        print("ssim: ", np.mean(all_ssim))
        
        return
    N_iters = N_iters * num_accum
    start = start * num_accum
    
    
    for i in range(start, N_iters):
        time0 = time.time()
        j = i//num_accum
        if use_batching:
            # Random from one image
            scene_i = int(np.random.choice(i_train)/views_per_scene)

            pose_indices = np.random.randint(scene_i * views_per_scene, scene_i* views_per_scene + num_train_per_scene, size = N_rand)
            ray_indices = np.random.randint(0, H*W, size = N_rand)
            
            batch = rays_rgb[pose_indices, ray_indices]

            batch = tf.transpose(batch, [1, 0, 2])

            batch_rays, target_s = batch[:2], batch[2]

            num_images = np.random.randint(*inp_img_range)

            p1 = np.random.randint(scene_i * views_per_scene, scene_i * views_per_scene + num_train_per_scene, size = num_images)
            print(num_train_per_scene,views_per_scene, scene_i)
            attention_images = images[p1]

            attention_poses = poses[p1,:3,:4]
        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            #scene associated with image
            scene_i = int(img_i/views_per_scene)

            target = read_im(images[img_i])
            pose = poses[img_i, :3, :4]
            render_pose = pose

            #N_rand is always not None
            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose)

                num_images = np.random.randint(*inp_img_range)

                attention_poses, attention_images = get_data(scene_i, num_images)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH),
                        tf.range(W//2 - dW, W//2 + dW),
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####

        #might have to do this one chunk at a time
        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, attention_images = attention_images,
                attention_poses = attention_poses, intrinsic = intrinsic, render_pose=render_pose,  **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)


        gradients = tape.gradient(loss, grad_vars)
        acm_loss += loss
        #Accumulate the gradients
        if acm_grads is None:
            acm_grads = [g/num_accum for g in gradients]
        if i % num_accum == 0:
            for p in range(len(acm_grads)):
                acm_grads[p]+=gradients[p]/num_accum
            optimizer.apply_gradients(zip(acm_grads, grad_vars))

            global_step.assign_add(1)
            acm_grads = None
            print("acm_loss", (acm_loss/num_accum).numpy())
            acm_loss = 0.
            acm_time = time.time()
        else:
            for p in range(len(acm_grads)):
                acm_grads[p]+=gradients[p]/num_accum
        dt = time.time() - time0

        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            def save_weights(net, prefix, i):
                path = os.path.join(
                    basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
                np.save(path, net.get_weights())
                print('saved weights at', path)
                if last_saved_dict[prefix].full():
                    os.remove(last_saved_dict[prefix].get())
                last_saved_dict[prefix].put(path)

            for k in models:
                if k is 'model' or k is 'model_fine':
                    save_weights(models[k].nerf_model, k, j)
            save_weights(models['optimizer'], 'optimizer', j)
            save_weights(models['attention_model'], 'attention_model',j)
            save_weights(models['unet_model'], 'unet_model',j)


        if i % args.i_print == 0 or i < 10:

            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            if i % args.i_img == 0:

                # All validation views during train come from scenes inside the training set
                img_i = np.random.choice(i_val)
                target = read_im(images[img_i])
                scene_i = int(img_i / views_per_scene)
                pose = poses[img_i, :3, :4]

                print("img_i")
                print(img_i)
                print(scene_i)
                print(pose)

                num_images = 2

                attention_poses, attention_images = get_data(scene_i, num_images)

                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                    attention_images = attention_images,
                                                    attention_poses = attention_poses, intrinsic=intrinsic,render_pose=pose,
                                                    **render_kwargs_test)


                psnr = mse2psnr(img2mse(rgb, target))
                print(expname, i, psnr.numpy(), tf.image.ssim(tf.convert_to_tensor(rgb),tf.convert_to_tensor(target),max_val=1.0).numpy(), global_step.numpy())

                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i == 0:
                    os.makedirs(testimgdir, exist_ok=True)
                set = '_in_train'
                imageio.imwrite(os.path.join(testimgdir, '{:06d}{}.png'.format(i,set)), to8b(rgb))
                imageio.imwrite(os.path.join(testimgdir, '{:06d}{}_target.png'.format(i,set)), to8b(target))
                imageio.imwrite(os.path.join(testimgdir, '{:06d}{}input1.png'.format(i,set)), to8b(attention_images[0]))
                imageio.imwrite(os.path.join(testimgdir, '{:06d}{}input2.png'.format(i,set)), to8b(attention_images[1]))

if __name__ == '__main__':
    train()
