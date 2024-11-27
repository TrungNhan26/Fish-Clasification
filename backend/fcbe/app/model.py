from tensorflow.keras.models import load_model

MODEL_PATH = "saved_model/fish_classification.h5"


def load_fish_model():
    # Load the model as usual
    model = load_model(MODEL_PATH, compile=False)
    return model