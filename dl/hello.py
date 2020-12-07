import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import render_template


app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

# load the model on server startup
dl_model = keras.models.load_model('./model_with_embeddings.h5')

print('HOLA MUNDO!!')

# agregar los parametros de la normalización

# agregar los diccionarios que tranforman vendor_id en vendor_idx

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test')
def test():
    # test model with one prediction
    x_test = np.array([0.23333333, 0.30666667, 0.1626506 , 0.25540275, 0.06644518, 0.02430556, 0.39631336, 0.47708082, 0.34108527, 0.08988764, 0.20289855, 0. , 1. ])
    vendor_idx_test = np.array([28])
    prediction = dl_model.predict([x_test.reshape(1,-1), vendor_idx_test.reshape(1,1)])
    prediction = np.where(prediction > 0.5, 1, 0)
    truth = 0

    print(dl_model.summary())
    return 'Mi primer modelo de DL: la verdadera categoria del vino es {} y la predicción es {}'.format(truth, prediction)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)

    # TODO
    # Use data to build x_test and vendor_idx_test
    # Use the model to get the prediction (0 or 1)
    # Be careful with NaNs, min, max, and data types
    # Transform the prediction and return it to the front-end

    data_x = np.zeros((13))
    data_vendor_id = np.array([28])

    prediction = dl_model.predict([data_x.reshape(1,-1), data_vendor_id.reshape(1,1)])

    result = None
    if prediction < 0.5:
        result = 'low quality wine'
    else:
        result = 'high quality wine'

    return {'quality': result}
