import joblib

def predict(data):
    model=joblib.load("./model/best_model.sav")
    return model.predict(data)