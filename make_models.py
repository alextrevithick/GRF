import tensorflow as tf

def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, image_features = 0):
    #dropouts = [7]
    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(image_features + input_ch + input_ch_views))
    inputs_features, inputs_pts, inputs_views = tf.split(inputs, [image_features, input_ch, input_ch_views], -1)
    inputs_features.set_shape([None, image_features])
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_features.shape, inputs_views.shape)
    outputs = tf.concat([inputs_features, inputs_pts], -1)
    for i in range(D):
        outputs = dense(W)(outputs)
        #outputs = tf.keras.layers.BatchNormalization()(outputs)
        if i in skips:
            outputs = tf.concat([inputs_features, inputs_pts, outputs], -1)
        #if i in dropouts:
        #    outputs = tf.keras.layers.Dropout(0.2)(outputs)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1):
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def init_nerf_model_no_pts(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, image_features = 0):
    #dropouts = [7]
    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(image_features + input_ch + input_ch_views))
    inputs_features, _, inputs_views = tf.split(inputs, [image_features, input_ch, input_ch_views], -1)
    inputs_features.set_shape([None, image_features])
    inputs_views.set_shape([None, input_ch_views])
    print("in the init no pts nerf")
    print(inputs_features.shape, inputs_views.shape)
    outputs = inputs_features
    for i in range(D):
        outputs = dense(W)(outputs)
        #outputs = tf.keras.layers.BatchNormalization()(outputs)
        if i in skips:
            outputs = tf.concat([inputs_features, outputs], -1)
        #if i in dropouts:
        #    outputs = tf.keras.layers.Dropout(0.2)(outputs)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1):
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class nerf_attention_model_obj(tf.keras.Model):
    def __init__(self, nerf_model, slot_att, embed_fn, embed_ln, embed_fn_2, embed_ln_2, coarse, num_samples = 64):
        super(nerf_attention_model_obj, self).__init__()
        self.nerf_model = nerf_model
        self.num_samples = num_samples
        self.embed_fn, self.embed_ln = embed_fn, embed_ln
        self.embed_fn_2, self.embed_ln_2 = embed_fn_2, embed_ln_2
        self.slot_att = slot_att
        self.coarse = coarse
    def call(self, inputs, training=None):
        # n_pts, embedding_len
        nerf_inputs = inputs[0]
        #indices: n_views, n_pts, 2 (int32); image_coords: n_views, n_pts, 2 (urounded, float32)
        local = inputs[1]

        pts = inputs[2]

        #take nerf embedded pts
        #embedded_pts = self.embed_fn(pts)
        embedded_pts = nerf_inputs[...,:self.embed_ln]
        embedded_pts = tf.broadcast_to(embedded_pts[None], (local.shape[0], local.shape[1], embedded_pts.shape[-1]) )


        attention_outputs = self.slot_att(tf.transpose(local, [1,0,2]), tf.transpose(embedded_pts, [1,0,2]))

        decoder_input = tf.concat([attention_outputs, nerf_inputs], -1)

        return self.nerf_model(decoder_input, training=training), decoder_input

def init_nerf_attention_model(nerf_model, slot_att, embed_fn, embed_ln,embed_fn_2, embed_ln_2, coarse, num_samples):
    return nerf_attention_model_obj(nerf_model,slot_att,embed_fn, embed_ln,embed_fn_2, embed_ln_2,coarse,num_samples)

class nerf_attention_model_obj_no_pts(tf.keras.Model):
    def __init__(self, nerf_model, slot_att, embed_fn, embed_ln, embed_fn_2, embed_ln_2, coarse, num_samples = 64):
        super(nerf_attention_model_obj_no_pts, self).__init__()
        print("in model init")
        self.nerf_model = nerf_model
        self.num_samples = num_samples
        self.embed_fn, self.embed_ln = embed_fn, embed_ln
        self.embed_fn_2, self.embed_ln_2 = embed_fn_2, embed_ln_2
        self.slot_att = slot_att
        self.coarse = coarse
    def call(self, inputs, training=None):
        # n_pts, embedding_len
        nerf_inputs = inputs[0]
        #indices: n_views, n_pts, 2 (int32); image_coords: n_views, n_pts, 2 (urounded, float32)
        local = inputs[1]

        pts = inputs[2]

        #change 10 to amount of nerf embedding
        embedded_pts = self.embed_fn(pts)
        embedded_pts = tf.broadcast_to(embedded_pts[None], (local.shape[0], local.shape[1], embedded_pts.shape[-1]) )


        attention_outputs = self.slot_att(tf.transpose(local, [1,0,2]), tf.transpose(embedded_pts, [1,0,2]))

        decoder_input = tf.concat([attention_outputs, nerf_inputs], -1)

        return self.nerf_model(decoder_input, training=training), decoder_input

def init_nerf_attention_model_no_pts(nerf_model, slot_att, embed_fn, embed_ln,embed_fn_2, embed_ln_2, coarse, num_samples):
    return nerf_attention_model_obj_no_pts(nerf_model,slot_att,embed_fn, embed_ln,embed_fn_2, embed_ln_2,coarse,num_samples)

def init_unet(embed_ln, dtype = 'shapenet', rotation_embed_ln = 0, use_globl = True, use_render_pose = True):
    if dtype == 'shapenet':
        return shapenet_unet(embed_ln, rotation_embed_ln = rotation_embed_ln, use_globl = use_globl, use_render_pose=use_render_pose)
    elif dtype == 'deepvoxels':
        return deepvoxels_unet(embed_ln)
    elif dtype == 'llff':
        return llff_unet(embed_ln, rotation_embed_ln = rotation_embed_ln, use_globl = use_globl, use_render_pose=use_render_pose)
    elif dtype == 'blender':
        return blender_unet(embed_ln, rotation_embed_ln = rotation_embed_ln, use_globl = use_globl, use_render_pose=use_render_pose)

def shapenet_unet(embed_ln, rotation_embed_ln = 0,  use_globl = True, use_render_pose = True):
    H,W = 128, 128
    num_embed = 3 if use_render_pose else 2
    num_rot = 2 if use_render_pose else 1
    input = tf.keras.layers.Input((H, W, num_embed * embed_ln + num_rot * rotation_embed_ln))
    embedded_rgb = input[...,:embed_ln]
    x_64 = tf.keras.layers.Conv2D(64, 7, 2, padding='same', activation='relu')(input)
    x_128 = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(x_64)
    x_256 = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation='relu')(x_128)
    x_512 = tf.keras.layers.Conv2D(512, 3, 2, padding='same', activation='relu')(x_256)

    if use_globl:
        globl = tf.keras.layers.Conv2D(128, 4, 4, padding='same', activation='relu')(x_512)
        globl = tf.reshape(globl, (-1, 512))
        globl = tf.keras.backend.repeat(globl, x_512.shape[2] * x_512.shape[1])
        globl = tf.reshape(globl, [-1, x_512.shape[1], x_512.shape[2], 512])
        globl = tf.concat([x_512, globl], -1)
        globl = tf.keras.layers.Dense(512, activation ='relu')(globl)
    else:
        print("no global")
    deepest = globl if use_globl else x_512
    x2_256 = tf.keras.layers.Conv2DTranspose(256, 3, dilation_rate=4, activation='relu')(deepest)

    x2_256 = tf.concat([x2_256, x_256], -1)
    x2_128 = tf.keras.layers.Conv2DTranspose(128, 3, dilation_rate=8, activation='relu')(x2_256)
    x2_128 = tf.concat([x2_128, x_128], -1)
    x2_64 = tf.keras.layers.Conv2DTranspose(64, 3, dilation_rate=16, activation='relu')(x2_128)
    x2_64 = tf.concat([x2_64, x_64], -1)
    local = tf.keras.layers.Conv2DTranspose(128, 3, dilation_rate=32, activation='relu')(x2_64)
    local = tf.concat([embedded_rgb, local], -1)

    return tf.keras.Model(input, local)

def deepvoxels_unet(embed_ln):
    H,W=512,512
    input = tf.keras.layers.Input((H, W, 3 * embed_ln))
    embedded_rgb = input[...,:embed_ln]
    x_64 = tf.keras.layers.Conv2D(64, 7, 2, padding='same', activation='relu')(input)
    x_128 = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(x_64)
    x_256 = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation='relu')(x_128)
    x_512 = tf.keras.layers.Conv2D(512, 3, 2, padding='same', activation='relu')(x_256)

    globl = tf.keras.layers.AveragePooling2D(4)(x_512)
    globl = tf.keras.layers.Conv2D(128, 4, 4, padding='same', activation='relu')(globl)
    globl = tf.reshape(globl, (-1, 512))
    globl = tf.keras.backend.repeat(globl, x_512.shape[2] * x_512.shape[1])
    globl = tf.reshape(globl, [-1, x_512.shape[1], x_512.shape[2], 512])
    globl = tf.concat([x_512, globl], -1)
    globl = tf.keras.layers.Dense(512, activation ='relu')(globl)

    x2_256 = tf.keras.layers.Conv2DTranspose(256, 3, dilation_rate=16, activation='relu')(globl)
    x2_256 = tf.concat([x2_256, x_256], -1)
    x2_128 = tf.keras.layers.Conv2DTranspose(128, 3, dilation_rate=32, activation='relu')(x2_256)
    x2_128 = tf.concat([x2_128, x_128], -1)
    x2_64 = tf.keras.layers.Conv2DTranspose(64, 3, dilation_rate=64, activation='relu')(x2_128)
    x2_64 = tf.concat([x2_64, x_64], -1)
    local = tf.keras.layers.Conv2DTranspose(128, 3, dilation_rate=128, activation='relu')(x2_64)
    local = tf.concat([embedded_rgb, local], -1)

    return tf.keras.Model(input, local)

def llff_unet(embed_ln, rotation_embed_ln = 0, use_globl = True, use_render_pose = True):
    H, W = 756, 1008
    num_embed = 3 if use_render_pose else 2
    num_rot = 2 if use_render_pose else 1
    input = tf.keras.layers.Input((H, W, num_embed * embed_ln + num_rot * rotation_embed_ln))
    input_padded = tf.keras.layers.ZeroPadding2D(((6,6), (0,0)))(input)
    embedded_rgb = input[...,:embed_ln]
    x_64 = tf.keras.layers.Conv2D(64, 7, 2, padding='same', activation='relu')(input_padded)
    x_128 = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(x_64)
    x_256 = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation='relu')(x_128)
    x_512 = tf.keras.layers.Conv2D(512, 3, 2, padding='same', activation='relu')(x_256)

    if use_globl:
        globl = tf.keras.layers.AveragePooling2D(8)(x_512)
        globl = tf.keras.layers.Conv2D(128, 4, 4, padding='same', activation='relu')(globl)
        globl = tf.reshape(globl, (-1, 512))
        globl = tf.keras.backend.repeat(globl, x_512.shape[2] * x_512.shape[1])
        globl = tf.reshape(globl, [-1, x_512.shape[1], x_512.shape[2], 512])
        globl = tf.concat([x_512, globl], -1)
        globl = tf.keras.layers.Dense(512, activation ='relu')(globl)

    deepest = globl if use_globl else x_512

    x2_256 = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(deepest)
    x2_256 = tf.concat([x2_256, x_256], -1)

    x2_128 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x2_256)
    x2_128 = tf.concat([x2_128, x_128], -1)

    x2_64 = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding = 'same')(x2_128)
    x2_64 = tf.concat([x2_64, x_64], -1)

    local = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding = 'same')(x2_64)
    local = local[:,6:-6]
    local = tf.concat([embedded_rgb, local], -1)
    local = tf.keras.layers.ZeroPadding2D( ((1,0), (0,1)) )(local)

    return tf.keras.Model(input, local)


def blender_unet(embed_ln, rotation_embed_ln = 0, use_globl = True, use_render_pose = True):
    H,W=800,800
    num_embed = 3 if use_render_pose else 2
    num_rot = 2 if use_render_pose else 1
    input = tf.keras.layers.Input((H, W, num_embed * embed_ln + num_rot * rotation_embed_ln))
    embedded_rgb = input[...,:embed_ln]
    x_64 = tf.keras.layers.Conv2D(64, 7, 2, padding='same', activation='relu')(input)
    x_128 = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(x_64)
    x_256 = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation='relu')(x_128)
    x_512 = tf.keras.layers.Conv2D(512, 3, 2, padding='same', activation='relu')(x_256)

    if use_globl:
        globl = tf.keras.layers.AveragePooling2D(5)(x_512)
        globl = tf.keras.layers.Conv2D(128, 5, 5, padding='same', activation='relu')(globl)
        globl = tf.reshape(globl, (-1, 512))
        globl = tf.keras.backend.repeat(globl, x_512.shape[2] * x_512.shape[1])
        globl = tf.reshape(globl, [-1, x_512.shape[1], x_512.shape[2], 512])
        globl = tf.concat([x_512, globl], -1)
        globl = tf.keras.layers.Dense(512, activation ='relu')(globl)

    deepest = globl if use_globl else x_512

    x2_256 = tf.keras.layers.Conv2DTranspose(256, 3, strides = 2, padding = 'same', activation='relu')(deepest)
    x2_256 = tf.concat([x2_256, x_256], -1)
    x2_128 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='same', activation='relu')(x2_256)
    x2_128 = tf.concat([x2_128, x_128], -1)
    x2_64 = tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same', activation='relu')(x2_128)
    x2_64 = tf.concat([x2_64, x_64], -1)
    local = tf.keras.layers.Conv2DTranspose(128, 3, strides=2,padding='same', activation='relu')(x2_64)
    local = tf.concat([embedded_rgb, local], -1)

    return tf.keras.Model(input, local)
