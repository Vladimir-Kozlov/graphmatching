import tensorflow as tf
from tensorflow import keras
import layers


def ZanSmi_feat_maps(**kwargs):
    img = keras.layers.Input(shape=(None, None, 3), name='image_input', tensor=kwargs.get('img'))

    vgg = keras.applications.vgg16.VGG16(input_shape=(None, None, 3), weights='imagenet',
                                         include_top=False, pooling='None')

    v = keras.models.Model(inputs=vgg.input, 
                           outputs=vgg.get_layer('block4_conv2').output)
    e = keras.models.Model(inputs=vgg.input, 
                           outputs=vgg.get_layer('block5_conv1').output)

    feat_map_vertex = v(img)
    feat_map_edge = e(img)

    return keras.models.Model(inputs=img, outputs=[feat_map_vertex, feat_map_edge])


def ZanSmi_transform_index(**kwargs):
    idx = keras.layers.Input(shape=(None, 2), dtype='int32', tensor=kwargs.get('keypoint_position'),
                             name='keypoints_coordinates_input')

    v = layers.IndexTransformationLayer(scale=(8, 8))
    e = layers.IndexTransformationLayer(scale=(16, 16))

    vertex_idx = v(idx)
    edge_idx = e(idx)

    return keras.models.Model(inputs=idx, outputs=[vertex_idx, edge_idx])


def ZanSmi_feat_extract(**kwargs):
    fmap = keras.layers.Input(shape=(None, None, None), name='feature_map', tensor=kwargs.get('feature_map'))
    idx = keras.layers.Input(shape=(None, 3), dtype='int32', name='keypoint_position_processed',
                             tensor=kwargs.get('keypoint_position_processed'))
    # alright, even I'm not autistic enough to make a new layer for this
    fmap_idx = keras.layers.Lambda(lambda x: tf.gather_nd(*x))([fmap, idx])
    return keras.models.Model(inputs=[fmap, idx], outputs=fmap_idx)


def ZanSmi_aff_vertex(**kwargs):
    # Builds vertex affinity matrix as proposed in Zanfir's paper
    # Input: vertex_feat_1, vertex_feat_2: feature tensors for vertices 
    #                                      of size [..., N1, feat_dim] and [..., N2, feat_dim] respectively
    # Output: keras Model that takes [vertex_feat_1, vertex_feat_2] and outputs affinity matrix Mp
    vertexfeat1 = keras.layers.Input(shape=(None, None), name='vertex_feat_1', tensor=kwargs.get('vertex_feat_1'))
    vertexfeat2 = keras.layers.Input(shape=(None, None), name='vertex_feat_2', tensor=kwargs.get('vertex_feat_2'))

    # ReLU added in order to produce nonnegative affinity
    Mp = keras.layers.ReLU()(layers.VertexAffinityLayer()([vertexfeat1, vertexfeat2]))
    return keras.models.Model(inputs=[vertexfeat1, vertexfeat2], outputs=Mp)


def ZanSmi_aff_edge(**kwargs):
    # Builds edge affinity matrix as proposed in Zanfir's paper
    # Input: edge_feat_1, edge_feat_2: feature tensors for vertices 
    #                                  of size [..., N1, feat_dim] and [..., N2, feat_dim] respectively
    #                                  edge feature is concatenation of source and target vertex feature vectors
    #        Warning: edge_feat_1 and edge_feat_2 are MANDATORY
    #        G1, G2: incidence matrices of size [..., N1, M1], [..., N2, M2] 
    #                (semantically; in reality, they have the same size due to padding)
    #                G[i, j] = 1 iff edge j starts in vertex i, otherwise G[i, j] = 0
    #        H1, H2: incidence matrices of size [..., N1, M1], [..., N2, M2] 
    #                (semantically; in reality, they have the same size due to padding)
    #                H[i, j] = 1 iff edge j endss in vertex i, otherwise H[i, j] = 0
    # Output: keras Model that takes [edge_feat_1, edge_feat_2, G1, G2, H1, H2] and outputs affinity matrix Mq
    edgefeat1 = keras.layers.Input(shape=(None, None), name='edge_feat_1', tensor=kwargs['edge_feat_1'])
    edgefeat2 = keras.layers.Input(shape=(None, None), name='edge_feat_2', tensor=kwargs['edge_feat_2'])
    G1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g1_input', tensor=kwargs.get('G1'))
    G2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g2_input', tensor=kwargs.get('G2'))
    H1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h1_input', tensor=kwargs.get('H1'))
    H2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h2_input', tensor=kwargs.get('H2'))

    # ReLU added in order to produce nonnegative affinity
    Mq = keras.layers.ReLU()(layers.EdgeAffinityLayer()([edgefeat1, edgefeat2, G1, G2, H1, H2]))
    return keras.models.Model(inputs=[edgefeat1, edgefeat2, G1, G2, H1, H2], outputs=Mq)


def ZanSmi_match(**kwargs):
    # Matching model: takes Mp, Mq, G1, G2, H1, H2 and performs spectral matching with Sinkhorn iterations
    # Input: Mp, Mq: vertex and edge affinity matrices respectively
    #                of size [..., N1, N2] and [..., M1, M2]
    #        G1, G2: incidence matrices of size [..., N1, M1], [..., N2, M2] 
    #                (semantically; in reality, they have the same size due to padding)
    #                G[i, j] = 1 iff edge j starts in vertex i, otherwise G[i, j] = 0
    #        H1, H2: incidence matrices of size [..., N1, M1], [..., N2, M2] 
    #                (semantically; in reality, they have the same size due to padding)
    #                H[i, j] = 1 iff edge j endss in vertex i, otherwise H[i, j] = 0
    # Output: keras Model that takes [Mp, Mq, G1, G2, H1, H2] and outputs soft matching matrix of size [..., N1, N2]
    Mp = keras.layers.Input(shape=(None, None), name='vertex_affinity', tensor=kwargs.get('vertex_affinity'))
    Mq = keras.layers.Input(shape=(None, None), name='edge_affinity', tensor=kwargs.get('edge_affinity'))
    G1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g1_input', tensor=kwargs.get('G1'))
    G2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g2_input', tensor=kwargs.get('G2'))
    H1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h1_input', tensor=kwargs.get('H1'))
    H2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h2_input', tensor=kwargs.get('H2'))

    power_max_iter = kwargs.get('power_max_iter', 100)
    power_eps_iter = kwargs.get('power_eps_iter', 1e-6)
    pi = layers.PowerIterationLayer(max_iter=power_max_iter, eps_iter=power_eps_iter,
                                    name='power_iteration_layer')([Mp, Mq, G1, G2, H1, H2])
    sinkhorn_max_iter = kwargs.get('sinkhorn_max_iter', 100)
    sinkhorn_eps_iter = kwargs.get('sinkhorn_eps_iter', 1e-6)
    sl = layers.SinkhornIterationLayer(max_iter=sinkhorn_max_iter, eps_iter=sinkhorn_eps_iter,
                                       name='sinkhorn_iteration_layer')(pi)
    return keras.models.Model(inputs=[Mp, Mq, G1, G2, H1, H2], outputs=sl, name='match_model')


def ZanSmi_full_model(**kwargs):
    img1 = keras.layers.Input(shape=(None, None, 3), name='image_1_input', tensor=kwargs.get('img1'))
    img2 = keras.layers.Input(shape=(None, None, 3), name='image_2_input', tensor=kwargs.get('img2'))
    m1 = ZanSmi_feat_maps()
    fmv1, fme1 = m1(img1)
    fmv2, fme2 = m1(img2)

    mm1 = keras.models.Model(inputs=[img1, img2], outputs=[fmv1, fmv2, fme1, fme2])

    idx1 = keras.layers.Input(shape=(None, 2), dtype='int32', tensor=kwargs.get('keypoint1'),
                              name='keypoints_coordinates_image_1_input')
    idx2 = keras.layers.Input(shape=(None, 2), dtype='int32', tensor=kwargs.get('keypoint2'),
                              name='keypoints_coordinates_image_2_input')
    m2 = ZanSmi_transform_index()
    idxv1, idxe1 = m2(idx1)
    idxv2, idxe2 = m2(idx2)

    mm2 = keras.models.Model(inputs=[idx1, idx2], outputs=[idxv1, idxv2, idxe1, idxe2])

    m3 = ZanSmi_feat_extract()
    featv1 = m3([fmv1, idxv1])
    feate1 = m3([fme1, idxe1])
    featv2 = m3([fmv2, idxv2])
    feate2 = m3([fme2, idxe2])

    mm3 = keras.models.Model(inputs=[img1, img2, idx1, idx2],
                             outputs=[featv1, featv2, feate1, feate2])
                             #outputs=[m3([x, y]) for (x, y) in zip(mm1.output, mm2.output)])

    m4 = ZanSmi_aff_vertex()

    z = mm3([img1, img2, idx1, idx2])
    
    Mp = m4([z[0], z[1]])
    mm4 = keras.models.Model(inputs=[img1, idx1, img2, idx2], outputs=Mp)

    G1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g1_input', tensor=kwargs.get('G1'))
    G2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_g2_input', tensor=kwargs.get('G2'))
    H1 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h1_input', tensor=kwargs.get('H1'))
    H2 = keras.layers.Input(shape=(None, None), name='incidence_matrix_h2_input', tensor=kwargs.get('H2'))
    m5 = ZanSmi_aff_edge(edge_feat_1=feate1, edge_feat_2=feate2, G1=G1, G2=G2, H1=H1, H2=H2)
    Mq = m5([z[2], z[3], G1, G2, H1, H2])
    
    mm5 = keras.models.Model(inputs=[img1, idx1, G1, H1, img2, idx2, G2, H2], outputs=Mq)

    m6 = ZanSmi_match(vertex_affinity=Mp, edge_affinity=Mq, G1=G1, G2=G2, H1=H1, H2=H2)

    Mp = mm4([img1, idx1, img2, idx2])
    Mq = mm5([img1, idx1, G1, H1, img2, idx2, G2, H2])
    X = m6([Mp, Mq, G1, G2, H1, H2])

    m = keras.models.Model(inputs=[img1, idx1, G1, H1, img2, idx2, G2, H2], outputs=X)
    return m


def deep_graph_matching_model():
    img1_input = keras.layers.Input(shape=(None, None, 3), dtype='float32')
    img2_input = keras.layers.Input(shape=(None, None, 3), dtype='float32')
    idx1_input = keras.layers.Input(shape=(None, 2), dtype='int32')
    idx2_input = keras.layers.Input(shape=(None, 2), dtype='int32')
    g1_input = keras.layers.Input(shape=(None, None), dtype='float32')
    g2_input = keras.layers.Input(shape=(None, None), dtype='float32')
    h1_input = keras.layers.Input(shape=(None, None), dtype='float32')
    h2_input = keras.layers.Input(shape=(None, None), dtype='float32')

    mnv2 = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(None, None, 3),
                                                       alpha=1.,
                                                       depth_multiplier=1, 
                                                       include_top=False, 
                                                       weights='imagenet', 
                                                       pooling=None)
    mnv2 = keras.models.Model(inputs=mnv2.input,
                              outputs=mnv2.get_layer('block_6_expand_relu').output)
    mnv2_1 = mnv2(img1_input)
    mnv2_2 = mnv2(img1_input)

    idx_vert = layers.IndexTransformationLayer(scale=(8, 8))
    idx_edge = layers.IndexTransformationLayer(scale=(8, 8))
    #idx_vert = keras.layers.Lambda(layers.idxtransform, arguments={'scale': 2**3})
    #idx_edge = keras.layers.Lambda(layers.idxtransform, arguments={'scale': 2**3})
    idxv_1 = idx_vert(idx1_input)
    idxe_1 = idx_edge(idx1_input)
    idxv_2 = idx_vert(idx2_input)
    idxe_2 = idx_edge(idx2_input)

    fmap_idx = keras.layers.Lambda(lambda x: tf.gather_nd(*x))
    featv_1 = fmap_idx([mnv2_1, idxv_1])
    feate_1 = fmap_idx([mnv2_1, idxe_1])
    featv_2 = fmap_idx([mnv2_2, idxv_2])
    feate_2 = fmap_idx([mnv2_2, idxe_2])

    # ReLU added in order to produce nonnegative affinity
    Mp = keras.layers.ReLU()(layers.VertexAffinityLayer()([featv_1, featv_2]))
    Mq = keras.layers.ReLU()(layers.EdgeAffinityLayer()([feate_1, feate_2, g1_input, g2_input, h1_input, h2_input]))
    pi = layers.PowerIterationLayer()([Mp, Mq, g1_input, g2_input, h1_input, h2_input])
    sl = layers.SinkhornIterationLayer()(pi)

    return keras.models.Model(inputs=[img1_input, idx1_input, g1_input, h1_input, img2_input, idx2_input, g2_input, h2_input], outputs=[sl])
    
