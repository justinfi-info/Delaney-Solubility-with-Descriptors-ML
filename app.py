import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from flask import Flask, render_template, request

from src.pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder="template")
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")

    try:
        # Get inputs from form (UPDATED for Delaney dataset)
        MolLogP = float(request.form.get("MolLogP", 0))
        MolWt = float(request.form.get("MolWt", 0))
        NumRotatableBonds = int(request.form.get("NumRotatableBonds", 0))
        AromaticProportion = float(request.form.get("AromaticProportion", 0))

        # Create custom data object
        data = CustomData(
            MolLogP=MolLogP,
            MolWt=MolWt,
            NumRotatableBonds=NumRotatableBonds,
            AromaticProportion=AromaticProportion
        )

        pred_df = data.get_data_as_dataframe()

        # Prediction
        result = PredictPipeline().predict(pred_df)[0]

        return render_template("index.html", results=round(float(result), 3))

    except Exception as e:
        return render_template("index.html", results=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
