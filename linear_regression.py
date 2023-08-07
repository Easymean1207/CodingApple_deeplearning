import tensorflow as tf

heights = [170, 175, 180, 165]
shoe_sizes = [270, 265, 265, 255]

height = 170
shoe_size = 260

a = tf.Variable(0.3)
b = tf.Variable(0.5)


def lossFunction():
    expectation = height * a + b
    # return tf.square(실제값 - 예측값)
    return tf.square(260 - expectation)


""" 경사하강법으로 구하는 함수 """
opt = tf.keras.optimizers.Adam(learning_rate=0.5)
# opt.minimize(손실함수, var_list[a,b] -> 경사하강법으로 업데이트할 weight variable 목록);

for i in range(300):
    opt.minimize(lossFunction, var_list=[a, b])
    print(a.numpy(), b.numpy())

print(height * a + b)
