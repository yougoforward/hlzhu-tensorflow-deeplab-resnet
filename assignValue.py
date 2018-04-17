from __future__ import print_function,division
import tensorflow as tf

w1 = tf.Variable(1, name='w1')
w2 = tf.Variable(2, name='w1')

print('w1.name=',w1.name)
print('w2.name=',w2.name)

# w3 = tf.get_variable(name='w3',initializer=1)
# w4 = tf.get_variable(name='w3',initializer=2)
w3 = tf.get_variable(name='w3',initializer=2)
print('w3.name=',w3.name)

with tf.variable_scope('scope1') as scope1:
    w1scope1 = tf.Variable(1, name='w1')
    print('w1scope1.name=', w1scope1.name)
    w4 = tf.get_variable(name='w4', initializer=1.0)
    print('w4.name=', w4.name)
    #scope1.reuse_variables() or tf.get_variable_scope().reuse_variables()
    tf.get_variable_scope().reuse_variables()
    w5 = tf.get_variable(name='w4', initializer=2.0)
    print('w5.name=', w5.name)

print('w1.name=', w1.name)

with tf.variable_scope('scope2', reuse=None) as scope2:
    w6 = tf.get_variable(name='w6', initializer=1.0)
    print('w6.name=', w6.name)
    w1 = tf.Variable(1, name='w1')
    print('w1.name=', w1.name)

print('w1.name=', w1.name)


with tf.variable_scope('scope2', reuse=True):
    w7 = tf.get_variable(name='w6', initializer=2.0)
    print('w7.name=', w7.name)




#create a Variable
w=tf.Variable(initial_value=[[1,2],[3,4]],dtype=tf.float32)
x=tf.Variable(initial_value=[[1,1],[1,1]],dtype=tf.float32,validate_shape=False)
y = tf.get_variable(name='y', initializer=1.0)
init_op=tf.global_variables_initializer()
update=tf.assign(x,[[1,2],[1,2]])
update2=tf.assign(y, 5.0)
tf.get_variable_scope().reuse_variables()
update2=tf.assign(tf.get_variable(name='y'), 5.0)

with tf.Session() as session:
    session.run(init_op)
    session.run(update)
    session.run(update2)
    x=session.run(x)
    y = session.run(y)
    print(x)
    print(y)