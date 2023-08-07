import tensorflow as tf

tensor = tf.constant([3, 4, 5])
tensor2 = tf.constant([6, 7, 8])

tensor3 = tf.constant([[1, 2, 3], [4, 5, 6]])
tensor3_1 = tf.constant([[7, 8], [9, 10], [11, 12]])

# 행렬의 곱 (dot product)
tensor_matmul = tf.matmul(tensor3, tensor3_1)
# print(tensor_matmul)

# 0만 담긴 텐서 -> tf.zeros(형태(tensor의 shape))
# shape의 해석은 뒤에서부터 하기!
tensor4 = tf.zeros([2, 2, 3])  # 3개 데이터를 담은 리스트를 2개 생성하고, 2개 셋트 생성
# print(tensor4)

# tensor 타입 변환
tensor_float = tf.constant([3, 4, 5], dtype=tf.float32)  # 타입 지정
tensor_float2 = tf.cast(tensor, tf.float32)  # 타입 변환
# print(tensor_float.dtype)
# print(tensor_float2)

# weight를 저장하고 싶을 때 -> Variable 만들기
w = tf.Variable(1.0)
# print(w.numpy())  # Variable에 저장된 값을 불러옴
w.assign(2.0)  # Variable에 새로운 값 할당
# print(w.numpy())
