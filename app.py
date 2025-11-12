from flask import Flask, render_template, request
import pickle, numpy as np, os, random
import requests

app = Flask(__name__)

# Load model and label encoder (prefer improved model if present)
MODEL_NAME = None
model = None
label_encoder = None
candidates = [
    ('model/crop_model_improved.pkl', 'model/label_encoder_improved.pkl'),
    ('model/crop_model.pkl', 'model/label_encoder.pkl')
]
for mpath, lpath in candidates:
    if os.path.exists(mpath) and os.path.exists(lpath):
        with open(mpath, 'rb') as f:
            model = pickle.load(f)
        with open(lpath, 'rb') as f:
            label_encoder = pickle.load(f)
        MODEL_NAME = os.path.basename(mpath)
        print(f"Loaded model: {mpath}")
        break
if model is None or label_encoder is None:
    raise FileNotFoundError('No model and label_encoder found in model/; expected crop_model(_improved).pkl and label_encoder(_improved).pkl')

# Weather API (OpenWeatherMap)
API_KEY = os.environ.get('OPENWEATHER_API_KEY', "YOUR_API_KEY_HERE")

# Language dictionary
translations = {
    'en': {
        'title': 'AI-Powered Crop Yield Prediction',
        'header': '🌾 Smart Crop Yield & Risk Analysis 🌾',
        'result_label': 'Prediction Result',
        'yield_label': 'Predicted Yield (in tons)',
        'risk_label': 'Risk Level',
        'back_button': 'Go Back',
        'submit_button': 'Predict Crop',
        'inputs': {
            'crop': 'Crop Name',
            'soil': 'Soil Type',
            'location': 'Location',
            'acres': 'Acres of Land'
        }
    },
    'te': {
        'title': 'AI ఆధారిత పంట దిగుబడి అంచనా వ్యవస్థ',
        'header': '🌾 స్మార్ట్ పంట దిగుబడి & ప్రమాద విశ్లేషణ 🌾',
        'result_label': 'ఫలితాలు',
        'yield_label': 'అంచనా దిగుబడి (టన్నులలో)',
        'risk_label': 'ప్రమాద స్థాయి',
        'back_button': 'తిరిగి వెళ్ళండి',
        'submit_button': 'పంట అంచనా వేయండి',
        'inputs': {
            'crop': 'పంట పేరు',
            'soil': 'మట్టి రకం',
            'location': 'ప్రాంతం',
            'acres': 'ఎకరాలు'
        }
    },
    'hi': {
        'title': 'एआई आधारित फसल उपज पूर्वानुमान प्रणाली',
        'header': '🌾 स्मार्ट फसल उपज और जोखिम विश्लेषण 🌾',
        'result_label': 'परिणाम',
        'yield_label': 'अनुमानित उत्पादन (टन में)',
        'risk_label': 'जोखिम स्तर',
        'back_button': 'वापस जाएं',
        'submit_button': 'फसल का पूर्वानुमान लगाएं',
        'inputs': {
            'crop': 'फसल का नाम',
            'soil': 'मिट्टी का प्रकार',
            'location': 'स्थान',
            'acres': 'भूमि (एकड़)'
        }
    }
}

# Available options for dropdowns (expanded lists)
SOIL_TYPES = [
    'Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky',
    'Sandy Loam', 'Silty Loam', 'Clay Loam', 'Loam'
]

CROP_TYPES = [
    'Rice', 'Wheat', 'Maize', 'Barley', 'Millet', 'Sorghum',
    'Soybean', 'Groundnut', 'Cotton', 'Sugarcane', 'Tea', 'Coffee',
    'Potato', 'Tomato', 'Onion', 'Cabbage', 'Cauliflower', 'Banana',
    'Mango', 'Grapes', 'Coconut', 'Pulses'
]

@app.route('/')
def home():
    lang = request.args.get('lang', 'en')
    text = translations.get(lang, translations['en'])
    return render_template('index.html', text=text, lang=lang,
                           soil_types=SOIL_TYPES, crop_types=CROP_TYPES)

@app.route('/predict', methods=['POST'])
def predict():
    lang = request.form.get('lang', 'en')
    text = translations.get(lang, translations['en'])

    # Accept selected option or custom text field
    crop_select = request.form.get('crop_select')
    crop_custom = request.form.get('crop')
    if crop_select and crop_select != 'Other':
        crop = crop_select
    else:
        crop = crop_custom or 'Unknown'

    soil_select = request.form.get('soil_select')
    soil_custom = request.form.get('soil')
    if soil_select and soil_select != 'Other':
        soil = soil_select
    else:
        soil = soil_custom or 'Unknown'
    location = request.form['location']
    acres = float(request.form['acres'])

    # --- Get weather data from API (with fallback) ---
    data = {}
    try:
        if API_KEY and API_KEY != "YOUR_API_KEY_HERE":
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        # If the API call fails (bad key, network, etc.) we'll fall back to simulated values below
        data = {}

    if data.get("main"):
        temperature = data['main'].get('temp', random.uniform(15, 30))
        humidity = data['main'].get('humidity', random.uniform(40, 90))
        rainfall = random.uniform(50, 300)  # Simulated rainfall value
        N, P, K, ph = random.randint(50,100), random.randint(30,60), random.randint(30,50), random.uniform(5.5,7.5)
    else:
        # No API key or failed request — use simulated weather & soil nutrient values so app still runs
        temperature = random.uniform(18, 30)
        humidity = random.uniform(40, 90)
        rainfall = random.uniform(50, 300)
        N, P, K, ph = random.randint(50,100), random.randint(30,60), random.randint(30,50), random.uniform(5.5,7.5)

    # --- Predict crop using ML model ---
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    predicted_crop = label_encoder.inverse_transform(model.predict(input_features))[0]

    # --- Yield estimation ---
    yield_estimate = round(acres * random.uniform(1.5, 3.0), 2)

    # --- Risk analysis ---
    if ph < 5.5 or ph > 8:
        risk = "High"
    elif soil.lower() in ["clay", "sandy"]:
        risk = "Moderate"
    else:
        risk = "Low"

    return render_template('result.html', crop=predicted_crop, yield_estimate=yield_estimate,
                           risk=risk, text=text, lang=lang, model_name=MODEL_NAME)

if __name__ == '__main__':
    app.run(debug=True)
