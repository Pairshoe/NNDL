from collections import defaultdict

import mnist_loader


def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    avgs = avg_darkness(training_data)
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
    print(f"{num_correct} of {len(test_data[1])} values correct")


def avg_darkness(training_data):
    digit_counts = defaultdict(int)
    darkness = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darkness[digit] += sum(image)
    avgs = defaultdict(float)
    for digit, n in digit_counts.items():
        avgs[digit] = darkness[digit] / n
    return avgs


def guess_digit(image, avgs):
    darkness = sum(image)
    distances = {k: abs(v - darkness) for k, v in avgs.items()}
    return min(distances, key=distances.get)


if __name__ == '__main__':
    main()
