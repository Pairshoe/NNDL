import mnist_loader
from sklearn import svm


def svm_benchmark():
    training_data, validation_data, test_data = mnist_loader.load_data()
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print(f"{num_correct} of {len(test_data[1])} values correct")


if __name__ == '__main__':
    svm_benchmark()
