from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

# directorio actual
print(os.getcwd())

# Flask app
app = Flask(__name__, template_folder='../templates')

# metodos get y post
@app.route('/', methods=['GET', 'POST'])
def main():
    # formulario
    if request.method == "POST":
        
        # cargar clasificador
        clf = joblib.load("output/model.pkl")
        
        # obtener valores de los inputs
        height = request.form.get("height")
        weight = request.form.get("weight")
        
        # inputs
        X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # prediccion
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    # template en templates/main.html
    return render_template("main.html", output = prediction)

# run app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000, debug=True)