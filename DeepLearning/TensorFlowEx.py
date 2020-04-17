import tensorflow as tf
print(tf.version)
with tf.compat.v1.Session() as session:
	a=tf.constant([2])
	b=tf.constant([3])
	c=tf.add(a,b)
	result=session.run(c)
	print(result)
string = tf.Variable('this is a string',tf.string)
number=tf.Variable(324,tf.int16)
print(number,string)
rank_tensor=tf.Variable(['test','ok','yes'],['hi','hi','hi'])
print(rank_tensor.shape)
print(tf.rank(rank_tensor))