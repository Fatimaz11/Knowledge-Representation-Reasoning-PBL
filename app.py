from flask import Flask, render_template, request
from fuzzy_weather_system import predict_comfort

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    level = None
    if request.method == 'POST':
        try:
            temp = float(request.form['temperature'])
            humid = float(request.form['humidity'])
            score, level = predict_comfort(temp, humid)
        except Exception as e:
            score, level = None, None

    return render_template('index.html', score=score, level=level)

if __name__ == '__main__':
    app.run(debug=True)
