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
    
