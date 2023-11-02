
from flask import Flask, render_template, request
import numpy as np
import pickle

# Make sure to provide the correct path to your pickle model file
model = pickle.load(open('model\\model.pkl', 'rb'))
# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def pred():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    col = ['weight_gain', 'cold_hands_and_feet', 'anxiety', 'irregular_sugar_level',
           'yellow_urine', 'acute_liver_failure', 'swelling_of_stomach', 'drying_and_tingling_lips', 'continuous_feel_of_urine',
           'internal_itching', 'polyuria', 'mood_swings', 'receiving_unsterile_injections',
           'stomach_bleeding', 'prominent_veins_on_calf', 'loss_of_smell', 'throat_irritation',
           'redness_of_eye', 'sinus_pressure', 'runny_nose', 'pain_during_bowel_movement',
           'pain_in_anal_region', 'cramps', 'bruising', 'enlarged_thyroid', 'brittle_nails',
           'swollen_extremities', 'slurred_speech', 'distention_of_abdomen', 'fluid_overload.1',
           'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'blister',
           'red_sore_around_nose', 'bloody_stool', 'swollen_blood_vessels', 'hip_joint_pain',
           'painful_walking', 'spinning_movements', 'altered_sensorium', 'toxic_look_(typhos)']

    if request.method == 'POST':
        input_symptoms = [str(x) for x in request.form.values()]
        # input_symptoms = ["anxiety","spinning_movement","slurred_speech","painful_walking","blister","skin_peeling"]
        print(input_symptoms)
        b = [0] * 54  # You have 42 symptoms in your list

        for x in range(0, 42):  # Adjust the range based on the number of symptoms
            for y in input_symptoms:
                if col[x] == y:
                    b[x] = 1

        b = np.array(b)
        b = b.reshape(1, -1)  # Reshape the input for prediction
        print(b)
        prediction = model.predict(b)
        prediction = prediction[0]
        print(prediction)
    return render_template('result.html', prediction_text="The probable diagnosis says it could be " + prediction)

if __name__ == "__main__":
    app.run(debug=True)


