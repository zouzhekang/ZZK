import tensorflow as tf

def correlation(img_left, img_right, max_disp=40):
    B, H, W, C = img_left.shape
    volume = tf.Variable(tf.ones([B, H, W, max_disp]))
    for i in range(max_disp):
        if i > 0:
            volume[:, :, i:,i].assign(tf.reduce_mean((img_left[:, :, i:, :] * img_right[:, :, :-i, :]),axis = 3))
        else:
            volume[:, :, :, i].assign(tf.reduce_mean((img_left[:, :, :, :] * img_right[:, :, :, :]),axis = 3))
            
    return volume
