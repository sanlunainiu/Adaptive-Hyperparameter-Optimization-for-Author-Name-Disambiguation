# GPU: cluster_size/count.py
from tensorflow.keras import backend as K


def l2Norm(x):
    return K.l2_normalize(x, axis=-1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    pos_dist = K.square(y_pred[:, 0]) 
    neg_dist = K.square(y_pred[:, 1])
    basic_loss = pos_dist - neg_dist + margin
    loss = K.maximum(basic_loss, K.constant(0))
    return K.mean(loss)



def accuracy(y_true, y_pred):
    return K.mean(K.less(y_pred[:, 0], y_pred[:, 1]))

