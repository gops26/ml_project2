from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
application = Flask(__name__)

app = application


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_datapoint", methods=["POST", "GET"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
                engine_size = request.form.get("engine_size"),
                cylinders=request.form.get("cylinders"),
                fuel_type=request.form.get("fuel_type"),
                fuel_consumption=request.form.get("fuel_consumption")
                )
        to_pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(to_pred_df)

        return render_template("home.html", result=result)

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)

