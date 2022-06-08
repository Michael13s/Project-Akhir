
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, static_url_path='/static')


@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("index.html")
    # elif request.method == 'POST':
    #     print(dict(request.form))
    #     height_features = dict(request.form).values()
    #     height_features = np.array([float(x) for x in height_features])
    #     model = joblib.load("height-weight-prediction-LR.pkl")
    #     height_features = ([height_features])
    #     print(height_features)
    #     result = model.predict(height_features)
       
    #     return render_template('index.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)


