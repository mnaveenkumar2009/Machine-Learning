# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
state=tf.Variable(0)
one=tf.constant(1)
newv=tf.add(state,one)
upd=tf.assign(state,newv)
init_op=tf.global_variables_initializer()
with tf.Session() as session:
    result=session.run(init_op)
    print(session.run(state))
    for _ in range(3):
        session.run(upd)
        print(session.run(state))

a=tf.placeholder(tf.float16)
b=a*2
result=tf.Session().run(b,feed_dict={a:3.5})
print result    