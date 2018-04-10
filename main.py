from mnist import MNIST
from multilayer_perceptron import MLP

mnist = MNIST(one_hot_encoding=True, z_score=True)

X_train = mnist.train_images
y_train = mnist.train_labels
X_test = mnist.test_images
y_test = mnist.test_labels

clf = MLP(hidden_layer_sizes=(500, 300), activation='swish', verbose=True)

clf.fit(X_train, y_train)

test_loss = clf._compute_loss(X_test, y_test)
test_acc = clf.score(X_test, y_test)
print('\nTest loss: {:.3}\tTest accuracy: {:.3}'.format(test_loss, test_acc))
