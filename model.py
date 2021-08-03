import pickle
from configparser import ConfigParser
from preprocess import process_data

config = ConfigParser()
config.read('./config.ini')
MODEL_PATH = config['DEFAULT']['MODEL_PATH']


with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


def predict_symptom(data, model=model):
    processed_data = process_data(data)
    predictions = model.predict(processed_data)
    return predictions


if __name__ == '__main__':
    print(predict_symptom())
