import joblib
import json

def crop_predict(N, P, K, temperature, humidity, ph, rainfall):
    import joblib  # Ensure joblib is imported

    model_path = r"C:\Users\ll010\Project-Takshak\functions\crop_predict.pkl"
    model = joblib.load(model_path)

    # Convert all inputs to float
    N = float(N)
    P = float(P)
    K = float(K)
    temperature = float(temperature)
    humidity = float(humidity)
    ph = float(ph)
    rainfall = float(rainfall)

    # Predict
    prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    return prediction


def yield_predict(state, crop_year, season, crop, area):
    model_path = r"C:\\Users\\ll010\\Project-Takshak\\functions\\yieldpred.pkl"
    mydict_path = r"C:\\Users\\ll010\\Project-Takshak\\functions\\yieldpred.txt"

    model = joblib.load(model_path)
    mydict = json.load(open(mydict_path))

    state = mydict[state]
    season = mydict[season]
    crop = mydict[crop]

    prediction = model.predict([[state, crop_year, season, crop, area]])
    return prediction

# Uncomment below to test
# state = 'NICOBARS'
# crop_year = 2000.0
# season = 'Kharif'
# crop = 'Arecanut'
# area = 1254.0
# print(yield_predict(state, crop_year, season, crop, area))
