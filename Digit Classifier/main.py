from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')

def main():
    # We set as_frame to False because MNIST is a dataset of images, which a Pandas DataFrame isn't ideal for.
    # Setting as_frame to False will return the data in the Batch as a NumPy array.
    mnist = fetch_openml('mnist_784', as_frame=False)

    x, y = mnist.data, mnist.target

    # This is only because the MNIST dataset formats the training set as the first 60k items and the testing set in the last 10k.
    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    y_train_5 = (y_train == '5')
    y_test_5 = (y_test == '5')

    some_digit = x[0]

    sgd_classifier = SGDClassifier(random_state=42)
    sgd_classifier.fit(x_train, y_train_5)

    print(sgd_classifier.predict([some_digit]))


if __name__ == "__main__":
    main()