import tensorflow as tf
from tensorflow import keras
import layers


def ZanSmi_full_model(**kwargs):
    img1 = keras.layers.Input(shape=(None, None, 3), name='image_1_input', tensor=kwargs.get('img1'))
    img2 = keras.layers.Input(shape=(None, None, 3), name='image_2_input', tensor=kwargs.get('img2'))
    vgg = keras.applications.vgg16.VGG16(input_shape=(None, None, 3), weights='imagenet',
                                         include_top=False, pooling='None')
    v = keras.models.Model(inputs=vgg.input, 
                           outputs=vgg.get_layer('block4_conv2').output)
    e = keras.models.Model(inputs=vgg.input, 
                           outputs=vgg.get_layer('block5_conv1').output)
    fmv1 = v(img1)
    fmv2 = v(img2)
    fme1 = e(img1)
    fme2 = e(img2)

    idx1 = keras.layers.Input(shape=(None, 2), dtype='int32', tensor=kwargs.get('keypoint1'),
                              name='keypoints_coordinates_image_1_input')
    idx2 = keras.layers.Input(shape=(None, 2), dtype='int32', tensor=kwargs.get('keypoint2'),
                              name='keypoints_coordinates_image_2_input')
    v = layers.IndexTransformationLayer(scale=(8, 8))
    e = layers.IndexTransformationLayer(scale=(16, 16))
    idxv1 = v(idx1)
    idxv2 = v(idx2)
    idxe1 = e(idx1)
    idxe2 = e(idx2)

    fmap_idx = keras.layers.Lambda(lambda x: tf.gather_nd(*x))
    featv1 = fmap_idx([fmv1, idxv1])
    featv2 = fmap_idx([fmv2, idxv2])
    feate1 = fmap_idx([fme1, idxe1])
    feate2 = fmap_idx([fme2, idxe2])

    Mp = keras.layers.ReLU()(layers.VertexAffinityLayer()([featv1, featv2]))
    G1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g1_input', tensor=kwargs.get('G1'))
    G2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g2_input', tensor=kwargs.get('G2'))
    H1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h1_input', tensor=kwargs.get('H1'))
    H2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h2_input', tensor=kwargs.get('H2'))
    Mq = keras.layers.ReLU()(layers.EdgeAffinityLayer()([feate1, feate2, G1, G2, H1, H2]))

    power_max_iter = kwargs.get('power_max_iter', 100)
    power_eps_iter = kwargs.get('power_eps_iter', 1e-6)
    pi = layers.PowerIterationLayer(max_iter=power_max_iter, eps_iter=power_eps_iter,
                                    name='power_iteration_layer')([Mp, Mq, G1, G2, H1, H2])
    sinkhorn_max_iter = kwargs.get('sinkhorn_max_iter', 100)
    sinkhorn_eps_iter = kwargs.get('sinkhorn_eps_iter', 1e-6)
    sl = layers.SinkhornIterationLayer(max_iter=sinkhorn_max_iter, eps_iter=sinkhorn_eps_iter,
                                       name='sinkhorn_iteration_layer')(pi)

    m = keras.models.Model(inputs=[img1, idx1, G1, H1, img2, idx2, G2, H2], outputs=sl)
    return m
    

def ZanSmi_mnv2_feat(**kwargs):
    img1 = keras.layers.Input(shape=(None, None, 3), name='image_1_input', tensor=kwargs.get('img1'))
    img2 = keras.layers.Input(shape=(None, None, 3), name='image_2_input', tensor=kwargs.get('img2'))
    mnv2 = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(None, None, 3), weights='imagenet', include_top=False, pooling=None)
    v = keras.models.Model(inputs=mnv2.input, 
                           outputs=mnv2.get_layer('block_6_expand_relu').output)
    e = keras.models.Model(inputs=mnv2.input, 
                           outputs=mnv2.get_layer('block_6_expand_relu').output)
    fmv1 = v(img1)
    fmv2 = v(img2)
    fme1 = e(img1)
    fme2 = e(img2)

    idx1 = keras.layers.Input(shape=(None, 2), dtype='int32', tensor=kwargs.get('keypoint1'),
                              name='keypoints_coordinates_image_1_input')
    idx2 = keras.layers.Input(shape=(None, 2), dtype='int32', tensor=kwargs.get('keypoint2'),
                              name='keypoints_coordinates_image_2_input')
    v = layers.IndexTransformationLayer(scale=8)
    e = layers.IndexTransformationLayer(scale=8)
    idxv1 = v(idx1)
    idxv2 = v(idx2)
    idxe1 = e(idx1)
    idxe2 = e(idx2)

    fmap_idx = keras.layers.Lambda(lambda x: tf.gather_nd(*x))
    featv1 = fmap_idx([fmv1, idxv1])
    featv2 = fmap_idx([fmv2, idxv2])
    feate1 = fmap_idx([fme1, idxe1])
    feate2 = fmap_idx([fme2, idxe2])

    Mp = keras.layers.ReLU()(layers.VertexAffinityLayer()([featv1, featv2]))
    G1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g1_input', tensor=kwargs.get('G1'))
    G2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g2_input', tensor=kwargs.get('G2'))
    H1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h1_input', tensor=kwargs.get('H1'))
    H2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h2_input', tensor=kwargs.get('H2'))
    Mq = keras.layers.ReLU()(layers.EdgeAffinityLayer()([feate1, feate2, G1, G2, H1, H2]))

    power_max_iter = kwargs.get('power_max_iter', 100)
    power_eps_iter = kwargs.get('power_eps_iter', 1e-6)
    pi = layers.PowerIterationLayer(max_iter=power_max_iter, eps_iter=power_eps_iter,
                                    name='power_iteration_layer')([Mp, Mq, G1, G2, H1, H2])
    sinkhorn_max_iter = kwargs.get('sinkhorn_max_iter', 100)
    sinkhorn_eps_iter = kwargs.get('sinkhorn_eps_iter', 1e-6)
    sl = layers.SinkhornIterationLayer(max_iter=sinkhorn_max_iter, eps_iter=sinkhorn_eps_iter,
                                       name='sinkhorn_iteration_layer')(pi)

    m = keras.models.Model(inputs=[img1, idx1, G1, H1, img2, idx2, G2, H2], outputs=sl)
    return m


def SMACNet(num_vertex, **kwargs):
    img1 = keras.layers.Input(shape=(None, None, 3), name='image_1_input', tensor=kwargs.get('img1'))
    img2 = keras.layers.Input(shape=(None, None, 3), name='image_2_input', tensor=kwargs.get('img2'))
    mnv2 = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(None, None, 3), weights='imagenet', include_top=False, pooling=None)
    v = keras.models.Model(inputs=mnv2.input, 
                           outputs=mnv2.get_layer('block_5_project').output,
                           name='vertex_feat_extract')
    e = keras.models.Model(inputs=mnv2.input, 
                           outputs=mnv2.get_layer('block_5_project').output,
                           name='edge_feat_extract')
    fmv1 = v(img1)
    fmv2 = v(img2)
    fme1 = e(img1)
    fme2 = e(img2)

    idx1 = keras.layers.Input(shape=(num_vertex[0], 2), dtype='int32', tensor=kwargs.get('keypoint1'),
                              name='keypoints_coordinates_image_1_input')
    idx2 = keras.layers.Input(shape=(num_vertex[1], 2), dtype='int32', tensor=kwargs.get('keypoint2'),
                              name='keypoints_coordinates_image_2_input')
    v = layers.IndexTransformationLayer(scale=8, name='vertex_index_transform')
    e = layers.IndexTransformationLayer(scale=8, name='edge_index_transform')
    idxv1 = v(idx1)
    idxv2 = v(idx2)
    idxe1 = e(idx1)
    idxe2 = e(idx2)

    fmap_idx = keras.layers.Lambda(lambda x: tf.gather_nd(*x), name='feature_map_index_layer')
    featv1 = fmap_idx([fmv1, idxv1])
    featv2 = fmap_idx([fmv2, idxv2])
    feate1 = fmap_idx([fme1, idxe1])
    feate2 = fmap_idx([fme2, idxe2])

    vmask1 = keras.layers.Input(shape=(num_vertex[0]), name='vertex_mask_1_input', tensor=kwargs.get('vmask1'))
    vmask2 = keras.layers.Input(shape=(num_vertex[1]), name='vertex_mask_2_input', tensor=kwargs.get('vmask2'))
    Mp = layers.VertexAffinityCosineLayer(name='vertex_affinity_layer')([featv1, vmask1, featv2, vmask2])

    G1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g1_input', tensor=kwargs.get('G1'))
    G2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g2_input', tensor=kwargs.get('G2'))
    H1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h1_input', tensor=kwargs.get('H1'))
    H2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h2_input', tensor=kwargs.get('H2'))
    makeedge = keras.layers.Lambda(lambda x: tf.concat([tf.matmul(x[1], x[0], transpose_a=True), 
                                                        tf.matmul(x[2], x[0], transpose_a=True)], axis=-1),
                                   name='make_edges_layer')
    e1 = makeedge([feate1, G1, H1])
    e2 = makeedge([feate2, G2, H2])
    Mq = layers.EdgeAffinityCosineLayer(name='edge_affinity_layer')([e1, e2])

    smac_max_iter = kwargs.get('smac_max_iter', 100)
    smac_eps_iter = kwargs.get('smac_eps_iter', 1e-6)
    smac = layers.SMACLayer(num_vertex, max_iter=smac_max_iter, eps_iter=smac_eps_iter, name='smac_layer')([Mp, Mq, G1, G2, H1, H2])

    m = keras.models.Model(inputs=[img1, idx1, vmask1, G1, H1, img2, idx2, vmask2, G2, H2], outputs=smac)
    return m

