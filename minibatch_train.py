import numpy as np
from optimizer import *
from mnist import load_mnist

# from train_neuralnet import TwoLayerNet
from backpropagation import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_loss_list = []
train_acc_list = []
test_acc_list = []


iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
optimizer = AddGrad()
# learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10,weight_init_std=0.1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # for key in ("W1", "b1", "W2", "b2"):
    #     # print("gradient attained")
    #     network.params[key] -= learning_rate * grad[key]

    optimizer.update(network.params, grad)

    # loss = network.loss(x_batch, t_batch)
    # train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc test acc | " + str(train_acc) + "," + str(test_acc))
