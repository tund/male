import tensorflow as tf
a = tf.Variable(1.0)
b = a + 1
c = 2.0
# sess = tf.InteractiveSession()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(b))
sess.run(tf.assign(a, c))
print(sess.run(b))
c = 3.0
sess.run(tf.assign(a, c))
print(sess.run(b))