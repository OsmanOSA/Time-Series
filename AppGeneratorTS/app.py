from flask import Flask, render_template, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils

app = Flask(__name__)

def generator(noise_size, label):
    # Load the model
    model = load_model("best_conditional_gan.h5")
    noise_random = tf.random.normal(shape=(noise_size, 100))
    label_input = tf.ones(shape=(noise_size)) * label
    
    gen = model.predict([noise_random, label_input])
    gen = gen.reshape(-1,)
    return gen

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_data():
    try:
        data = request.get_json()
        
        # Validation des données reçues
        if not data or 'labels' not in data or 'noise_sizes' not in data or 'seasons' not in data:
            return jsonify({
                'error': 'Données invalides. Les labels, noise_sizes et seasons sont requis.'
            }), 400
            
        labels = data['labels']
        noise_sizes = data['noise_sizes']
        seasons = data['seasons']
        
        # Validation du format des données
        if len(labels) != len(noise_sizes) or len(labels) != len(seasons):
            return jsonify({
                'error': 'Format invalide. Le nombre de labels, tailles et saisons doit être identique.'
            }), 400
        
        # Vérification des valeurs négatives ou nulles
        if any(not isinstance(size, (int, float)) or size <= 0 for size in noise_sizes):
            return jsonify({
                'error': 'Les valeurs doivent être des nombres strictement positifs.'
            }), 400
            
        # Vérification des valeurs trop grandes
        if any(size > 100 for size in noise_sizes):
            return jsonify({
                'error': 'Les valeurs ne doivent pas dépasser 100.'
            }), 400
        
        # Création de la figure avec le nombre exact de sous-graphiques
        fig = make_subplots(rows=len(seasons), cols=1, subplot_titles=seasons, shared_yaxes=True)
        
        for i in range(len(labels)):
            generated_data = generator(noise_sizes[i], labels[i]) * 382.0
            fig.append_trace(
                go.Scatter(
                    y=generated_data.tolist(),
                    mode="lines",
                    showlegend=False
                ),
                row=i+1,
                col=1
            )
        
        fig.update_layout(
            title="Données Générées par CGAN",
            xaxis_title="Step",
            yaxis_title="Valeur générée",
            height=400 * len(labels),  # Hauteur adaptative
            width=1000
        )
        
        # Convertir la figure en JSON compatible
        fig_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
        return fig_json

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
