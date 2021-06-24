import numpy as np

def predict(w, X):

    predictions = []
    for xi in X:
        net_input = np.dot(xi, w[1:]) + w[0]
        prediction = 1 if (net_input >= 0) else -1
        predictions.append(prediction)
    return predictions

