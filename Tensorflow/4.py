# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,6)
# X=np.arange(0,5,0.1) #like linspace
# #print X
# a=1
# b=0
# Y=a*X+b
# plt.plot(X,Y)
# plt.xlabel('dep')
# plt.ylabel('ind')
# plt.show()
x_data=np.random.rand(100).astype(np.float32)
y_data=2*x_data+3
y_data=np.vectorize(lambda y:y+np.random.normal(loc=0.0,scale=0.1))(y_data)
print zip(x_data,y_data)[0:5]

a=tf.Variable(1.0)
b=tf.Variable(0.5)
y=a*x_data+b
loss=tf.reduce_mean(tf.square(y_data-y))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
train_data=[]
for step in range(100):
    evals=sess.run([train,a,b])[1:]
    if step%5==0:
        print step,evals
        train_data.append(evals)

converter=plt.colors
cr,cg,cb=(1,1,0)
for f in train_data:
    cb+=1.0/len(train_data)
    cg-=1.0/len(train_data)
    if cb>1.0: cb=1.0
    if cg<0.0: cg=0.0
    [a,b]=f
    f_y=np.vectorize(lambda x:a*x+b)(x_data)
    line=plt.plot(x_data,f_y)
    plt.setp(line,color=(cr,cg,cb))

plt.plot(x_data,y_data,'ro')
plt.show()