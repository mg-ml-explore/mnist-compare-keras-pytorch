from . import data_utils
from . import model
from . import train_test_utils

def run():
    x_train, y_train = data_utils.get_training_samples()

    model = model.get_model()
    model.summary()

    train_test_utils.train_model(model, x_train, y_train)

    x_test, y_test = data_utils.get_testing_samples()
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

if __name__ == "__main__":
    run()
