import sys
import os

from flask import Flask,render_template,request,jsonify

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model')))
from model import StockForecaster

frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Frontend'))

app = Flask(__name__,
    template_folder=frontend_path,
    static_folder=os.path.join(frontend_path))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.form['symbol'].strip().upper()
        period = request.form['period'].strip()
        predict_days = int(request.form['predict_days'])
        epochs = int(request.form['epochs'])

        # Initialize and run model
        forecaster = StockForecaster(symbol, period)
        data = forecaster.fetch_data()
        if data is None:
            return jsonify({'error': 'Could not fetch stock data. Check symbol and period.'})

        results = forecaster.train_model(epochs=epochs)
        future = forecaster.predict_future(days=predict_days)

        # Extract prediction summary
        current_price = data['Close'].iloc[-1]
        future_price = future['Predicted_Price'].iloc[-1]
        change = future_price - current_price
        percent = (change / current_price) * 100

        return jsonify({
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(future_price, 2),
            'change': round(change, 2),
            'percent': round(percent, 2),
            'future': future.to_dict(orient='records')  
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
