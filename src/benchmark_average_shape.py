from collections import OrderedDict

import mnist_loader

import numpy as np


def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    avgs = avg_shape(training_data)
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
    print(f"{num_correct} of {len(test_data[1])} values correct")


def avg_shape(training_data):
    avgs = {}
    nums = {}
    for image, digit in zip(training_data[0], training_data[1]):
        if avgs.get(digit) is not None:
            avgs[digit] = [x + y for x, y in zip(avgs[digit], image)]
            nums[digit] += 1
        else:
            avgs[digit] = image
            nums[digit] = 1
    for digit in avgs:
        avgs[digit] = [value / nums[digit] for value in avgs[digit]]
    return OrderedDict(sorted(avgs.items()))


def guess_digit(image, avgs):
    return np.argmax([np.dot(image, avgs[digit]) for digit in avgs])


if __name__ == '__main__':
    main()
