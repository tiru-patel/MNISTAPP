from flask import Flask, render_template,request
import requests
import numpy as np
from PIL import Image
import json
app = Flask(__name__)
model_uri = "http://2ddcb151-1b9f-4b76-859f-aa25efaf6d39.eastus.azurecontainer.io/score"
@app.route("/")
@app.route("/home", methods=['POST'])
def home():
    return render_template('index.html')

@app.route("/uploader", methods = ['GET','POST'])
def upload_file():
    if request.method == "POST":
        img = Image.open(request.files['file'].stream).convert("L")
        img = img.resize((28,28))

        img2arr = np.array(img)
        img2arr = img2arr / 255.0
        img2arr = img2arr.reshape(1,-1)

        #x = img2arr.shape
        test = json.dumps({"data": img2arr.tolist()})
        #test = bytes(test, encoding='utf8')
        #input_data = "{\"data\":[" + str(list(test)) + "]}"
        headers = {'Content-Type':'application/json'}

        resp = requests.post(model_uri,test,headers=headers)
        pred = resp.text

        return render_template("predict.html", data=pred)

if __name__ == "__main__":
    app.run(debug=True)