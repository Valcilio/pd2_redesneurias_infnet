import os
from flask import Flask, request, jsonify
from model_pipeline import ModelPipeline

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if data: 
            if isinstance(data, dict): 
                pass
        
        # Call the function from model_pipeline.py
        df = ModelPipeline().structure(data)
        df1 = ModelPipeline().preprocessing(df)
        result = ModelPipeline().predict(df1, 'model.keras')
        
        # Return the result as JSON
        return jsonify({'prediction': int(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port, debug=True)