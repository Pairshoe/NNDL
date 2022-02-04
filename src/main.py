import mnist_loader
import network
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network.Network([784, 15, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, evaluation_data=validation_data,
        monitor_training_cost=True, monitor_training_accuracy=True,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True)
