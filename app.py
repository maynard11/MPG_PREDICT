from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from category_encoders import HashingEncoder
import os

try:
    model = joblib.load("mpg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
except FileNotFoundError:
    print("[v0] Model files not found! Training model...")
    import subprocess
    subprocess.run(["python", "train_model.py"], check=True)
    model = joblib.load("mpg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

app = Flask(__name__)

def interpret_mpg(mpg_value):
    if mpg_value < 20:
        return "Poor fuel efficiency - typical for larger vehicles, trucks, and performance cars"
    elif 20 <= mpg_value < 30:
        return "Moderate fuel efficiency - common for midsize sedans and smaller SUVs"
    elif 30 <= mpg_value < 40:
        return "Good fuel efficiency - typical for compact cars and efficient sedans"
    else:
        return "Excellent fuel efficiency - common for hybrids, electric vehicles, and very efficient compact cars"

# Enhanced HTML template with better contrast

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MPG RACER - ARCADE EDITION</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Press+Start+2P&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        @keyframes scanlines {
            0% { transform: translateY(0); }
            100% { transform: translateY(10px); }
        }
        
        @keyframes flicker {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.97; }
        }
        
        @keyframes pixelGlow {
            0%, 100% { text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #0099ff; }
            50% { text-shadow: 0 0 20px #00ff00, 0 0 30px #00ff00, 0 0 40px #0099ff; }
        }
        
        @keyframes arcade {
            0%, 100% { color: #00ff00; }
            50% { color: #0099ff; }
        }
        
        @keyframes speedometer {
            0% { transform: rotate(-90deg); }
            100% { transform: rotate(90deg); }
        }
        
        body {
            font-family: 'Press Start 2P', cursive;
            background: #000;
            color: #00ff00;
            background-image: 
                repeating-linear-gradient(
                    0deg,
                    rgba(0, 255, 0, 0.03),
                    rgba(0, 255, 0, 0.03) 1px,
                    transparent 1px,
                    transparent 2px
                );
            animation: scanlines 8s linear infinite, flicker 0.15s infinite;
            min-height: 100vh;
            padding: 20px;
            overflow-x: hidden;
        }
        
        .arcade-container {
            max-width: 1000px;
            margin: 0 auto;
            border: 6px solid;
            border-image: linear-gradient(45deg, #00ff00, #0099ff, #ff00ff, #00ffff) 1;
            background: #0a0a0a;
            box-shadow: 
                inset 0 0 30px rgba(0, 255, 0, 0.2),
                0 0 50px rgba(0, 255, 0, 0.3),
                0 0 100px rgba(0, 153, 255, 0.2);
            padding: 30px;
        }
        
        .game-header {
            text-align: center;
            margin-bottom: 40px;
            animation: pixelGlow 2s ease-in-out infinite;
        }
        
        .game-title {
            font-size: 2.5rem;
            margin-bottom: 10px;
            color: #00ff00;
            letter-spacing: 4px;
            text-shadow: 0 0 20px #00ff00, 0 0 40px #0099ff;
            animation: arcade 1s ease-in-out infinite;
        }
        
        .pixel-car {
            font-size: 3rem;
            margin: 20px 0;
            animation: bounce 0.5s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .subtitle {
            font-size: 0.8rem;
            color: #0099ff;
            text-shadow: 0 0 10px #0099ff;
            letter-spacing: 2px;
        }
        
        .level-indicator {
            display: inline-block;
            background: #00ff00;
            color: #000;
            padding: 8px 16px;
            margin: 10px 0;
            font-size: 0.7rem;
            border: 2px solid #00ff00;
            box-shadow: inset 0 0 10px rgba(0, 255, 0, 0.3);
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 30px 0;
        }
        
        .form-group {
            position: relative;
        }
        
        .form-label {
            display: block;
            font-size: 0.7rem;
            margin-bottom: 8px;
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff;
            letter-spacing: 1px;
            background: linear-gradient(90deg, transparent,transparent);
            padding: 4px 0;
        }
        
        input, select {
            width: 100%;
            padding: 12px;
            background: #1a1a1a;
            border: 3px solid #00ff00;
            color: #00ff00;
            font-family: 'Press Start 2P', cursive;
            font-size: 0.65rem;
            box-shadow: inset 0 0 10px rgba(0, 255, 0, 0.2), 0 0 10px rgba(0, 255, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #0099ff;
            box-shadow: 
                inset 0 0 15px rgba(0, 153, 255, 0.4),
                0 0 20px rgba(0, 153, 255, 0.6);
            background: #2a2a3a;
        }
        
        input::placeholder {
            color: #00ff00;
            opacity: 0.3;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }
        
        .btn {
            flex: 1;
            padding: 15px;
            background: linear-gradient(135deg, #00ff00, #00ffff);
            color: #000;
            border: 3px solid #00ff00;
            font-family: 'Press Start 2P', cursive;
            font-size: 0.75rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.5), inset 0 0 10px rgba(255, 255, 255, 0.2);
        }
        
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(0, 153, 255, 0.😎, inset 0 0 15px rgba(255, 255, 255, 0.4);
            border-color: #0099ff;
        }
        
        .btn:active {
            transform: scale(0.98);
        }
        
        .info-box {
            background: linear-gradient(135deg, #1a1a3a, #2a1a3a);
            border: 2px dashed #ff00ff;
            padding: 15px;
            margin: 20px 0;
            font-size: 0.65rem;
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff;
            line-height: 1.8;
        }
        
        .range-info {
            font-size: 0.6rem;
            color: #ffff00;
            margin-top: 5px;
            text-shadow: 0 0 10px #ffff00;
        }
        
        @media (max-width: 768px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .game-title {
                font-size: 1.8rem;
            }
            .button-group {
                flex-direction: column;
            }
        }
    </style>
</head>

<body>
<!-- AUDIO ELEMENT -->
    <audio id="bgMusic" autoplay loop volume="0.1">
        <source src="https://orangefreesounds.com/wp-content/uploads/2022/06/Arcade-background-music-retro-style.mp3" type="audio/mpeg">
    </audio>

    <!-- MUSIC TOGGLE BUTTON -->
    <button id="musicToggle" style="position: fixed; top: 20px; right: 20px; z-index: 1000; padding: 10px 15px; background: #ff00ff; color: #000; border: 3px solid #00ff00; border-radius: 5px; cursor: pointer; font-family: 'Press Start 2P'; font-size: 0.8em; box-shadow: 0 0 20px #ff00ff;">
        🎵 MUSIC: ON
    </button>

    <!-- Improved music script: added .catch() for autoplay restrictions -->
    <script>
        const audio = document.getElementById('bgMusic');
        const musicToggle = document.getElementById('musicToggle');
        
        // Try to autoplay, handle browser restrictions
        audio.play().catch(() => {
            console.log('Autoplay prevented - user must click to enable');
        });
        
        musicToggle.addEventListener('click', () => {
            if (audio.paused) {
                audio.play();
                musicToggle.textContent = '🎵 MUSIC: ON';
                musicToggle.style.background = '#ff00ff';
            } else {
                audio.pause();
                musicToggle.textContent = '🔇 MUSIC: OFF';
                musicToggle.style.background = '#888888';
            }
        });
    </script>

    <div class="arcade-container">
        <div class="game-header">
            <div class="pixel-car">🏎️</div>
            <div class="game-title">MPG PREDICTOR</div>
            <div class="subtitle">⚡ ARCADE EDITION ⚡</div>
            <div class="level-indicator">LEVEL 1: VEHICLE SETUP</div>
        </div>
        
        <div class="info-box">
            MISSION: Configure your vehicle to predict fuel efficiency!<br>
            <br>
            Fill in all fields and press START to calculate your MPG rating.
        </div>
        
        <form action="/predict" method="post">
            <div class="form-grid">
                <div class="form-group">
                    <label class="form-label">⚙️ Engine Size</label>
                    <input type="number" name="Engine_Size" step="0.1" min="0.5" max="8.0" required
                           placeholder="2.0">
                           <br>
                           <br>
                    <div class="range-info">Range: 0.5 - 8.0L</div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">⚡ Engine Cylinders</label>
                    <input type="number" name="Engine_Cylinders" min="2" max="16" required
                           placeholder="4">
                           <br>
                           <br>
                    <div class="range-info">Range: 2 - 16</div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">🛣️ Drive Type</label>
                    <select name="Drive_Type" required>
                        <option value="">SELECT...</option>
                        <option value="FWD">FWD - Front Wheel</option>
                        <option value="RWD">RWD - Rear Wheel</option>
                        <option value="AWD">AWD - All Wheel</option>
                        <option value="4WD">4WD - Four Wheel</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">⛽ Fuel Type</label>
                    <select name="Fuel_Type" required>
                        <option value="">SELECT...</option>
                        <option value="Gasoline">Gasoline</option>
                        <option value="Diesel">Diesel</option>
                        <option value="Hybrid">Hybrid</option>
                        <option value="Electric">Electric</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">🚗 Vehicle Type</label>
                    <select name="Vehicle Class/Type" required>
                        <option value="">SELECT...</option>
                        <option value="Sedan">Sedan</option>
                        <option value="SUV">SUV</option>
                        <option value="Truck">Truck</option>
                        <option value="Van">Van</option>
                        <option value="Coupe">Coupe</option>
                        <option value="Hatchback">Hatchback</option>
                        <option value="Convertible">Convertible</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">🏷️ Car Brand</label>
                    <input type="text" name="Car_Brand" required placeholder="Toyota">
                    <br>
                           <br>
                    <div class="range-info">e.g., Toyota, Ford, BMW</div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">📅 Model Year</label>
                    <input type="number" name="Model_Year" min="1990" max="2024" required
                           placeholder="2023">
                           <br>
                           <br>
                    <div class="range-info">Range: 1990 - 2024</div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">🪣 Fuel Capacity</label>
                    <input type="number" name="Fuel_Capacity" step="1" min="1000" max="5000" required
                           placeholder="3112">
                           <br>
                           <br>
                    <div class="range-info">Units: 2000-3500</div>
                </div>
            </div>
            
            <div class="button-group">
                <button type="submit" class="btn">START - PREDICT!</button>
            </div>
        </form>
    </div>
</body>
</html>
"""
RESULT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
    <style>
        @keyframes scanlines {
            0% { transform: translateY(0); }
            100% { transform: translateY(4px); }
        }
        @keyframes flicker {
            0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% { opacity: 1; }
            20%, 24%, 55% { opacity: 0.95; }
        }
        @keyframes pulse-mpg {
            0%, 100% { transform: scale(1); text-shadow: 0 0 20px #ffff00, 0 0 40px #ffff00, 0 0 60px #ff00ff; }
            50% { transform: scale(1.05); text-shadow: 0 0 30px #ffff00, 0 0 60px #ffff00, 0 0 90px #ff00ff; }
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Press Start 2P', cursive;
            background: #000;
            color: #00ff00;
            padding: 20px;
            min-height: 100vh;
            overflow-x: hidden;
        }
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background:
                repeating-linear-gradient(
                    0deg,
                    rgba(0, 0, 0, 0.3),
                    rgba(0, 0, 0, 0.3) 1px,
                    transparent 1px,
                    transparent 2px
                );
            pointer-events: none;
            animation: scanlines 0.15s linear infinite;
            z-index: 1;
        }
        .container {
            max-width: 700px;
            margin: 0 auto;
            background: #1a1a1a;
            border: 4px solid #ffff00;
            padding: 40px;
            box-shadow: 0 0 30px #ffff00, inset 0 0 20px rgba(255, 255, 0, 0.1);
            position: relative;
            z-index: 2;
            text-align: center;
        }
        .header {
            margin-bottom: 40px;
            border-bottom: 3px dashed #ffff00;
            padding-bottom: 20px;
        }
        .header h1 {
            font-size: 1.3em;
            color: #ffff00;
            text-shadow: 0 0 15px #ffff00, 0 0 30px #ff00ff;
            animation: flicker 0.15s infinite;
            letter-spacing: 2px;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 0.7em;
            color: #ff00ff;
            text-shadow: 0 0 5px #ff00ff;
            letter-spacing: 1px;
        }
        .mpg-value {
            font-size: 3.5em;
            color: #ffff00;
            background: #000;
            border: 3px solid #ffff00;
            padding: 30px;
            margin: 40px 0;
            text-shadow: 0 0 20px #ffff00, 0 0 40px #ffff00, 0 0 60px #ff00ff;
            box-shadow: 0 0 30px #ffff00, inset 0 0 15px rgba(255, 255, 0, 0.2);
            animation: pulse-mpg 1.5s ease-in-out infinite;
            letter-spacing: 2px;
        }
        .interpretation {
            font-size: 0.65em;
            color: #00ff00;
            text-shadow: 0 0 10px #00ff00;
            line-height: 1.6;
            background: rgba(0, 255, 0, 0.05);
            border: 2px solid #00ff00;
            padding: 20px;
            margin: 30px 0;
            letter-spacing: 0.5px;
        }
        .btn {
            background: #000;
            color: #ffff00;
            border: 3px solid #ffff00;
            padding: 15px 30px;
            font-family: 'Press Start 2P', cursive;
            font-size: 0.7em;
            cursor: pointer;
            text-shadow: 0 0 10px #ffff00;
            box-shadow: 0 0 10px #ffff00;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            letter-spacing: 1px;
        }
        .btn:hover {
            background: #ffff00;
            color: #000;
            box-shadow: 0 0 20px #ffff00, inset 0 0 10px rgba(0, 0, 0, 0.5);
            transform: scale(1.08);
        }
        .btn:active {
            transform: scale(0.98);
        }
        .game-over-message {
            font-size: 0.8em;
            color: #ff00ff;
            text-shadow: 0 0 10px #ff00ff;
            margin-bottom: 30px;
            letter-spacing: 1px;
        }
        /* Added music toggle button styling */
        #musicToggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            padding: 10px 15px;
            background: #ff00ff;
            color: #000;
            border: 3px solid #00ff00;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Press Start 2P';
            font-size: 0.8em;
            box-shadow: 0 0 20px #ff00ff;
            transition: all 0.3s ease;
        }
        #musicToggle:hover {
            box-shadow: 0 0 30px #ff00ff, 0 0 15px #00ff00;
        }
    </style>
</head>
<body>
    <!-- Added audio elements for result page music -->
    <!-- Victory/Results background music -->
    <audio id="bgMusic" loop>
        <source src="https://orangefreesounds.com/wp-content/uploads/2024/07/Retro-video-game-music.mp3" type="audio/mpeg">
    </audio>
    
    <!-- Victory fanfare sound effect (plays once on page load) -->
    <audio id="victorySound">
        <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj==" type="audio/wav">
    </audio>

    <!-- Music toggle button -->
    <button id="musicToggle">
        🎵 MUSIC: ON
    </button>

    <!-- Added music control script -->
    <script>
        const bgMusic = document.getElementById('bgMusic');
        const victorySound = document.getElementById('victorySound');
        const musicToggle = document.getElementById('musicToggle');
        
        // Play victory sound immediately when page loads
        window.addEventListener('load', () => {
            victorySound.play().catch(err => console.log('Victory sound autoplay prevented'));
            setTimeout(() => {
                bgMusic.play().catch(err => console.log('Background music autoplay prevented - user must click'));
            }, 500); // Delay background music slightly after victory sound
        });
        
        // Music toggle functionality
        musicToggle.addEventListener('click', () => {
            if (bgMusic.paused) {
                bgMusic.play();
                musicToggle.textContent = '🎵 MUSIC: ON';
                musicToggle.style.background = '#ff00ff';
            } else {
                bgMusic.pause();
                musicToggle.textContent = '🔇 MUSIC: OFF';
                musicToggle.style.background = '#888888';
            }
        });
    </script>

    <div class="container">
        <div class="header">
            <h1>★ MISSION COMPLETE ★</h1>
            <p>Your MPG Score</p>
        </div>
        <div class="game-over-message">
            >>> EFFICIENCY RATING <<<
        </div>
        <div class="mpg-value">{{ mpg }} MPG</div>
        <div class="interpretation">
            {{ interpretation }}
        </div>
        <button class="btn" onclick="window.location.href='/'">★ TRY AGAIN ★</button>
    </div>
</body>
</html>"""
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            'Engine_Size': float(request.form['Engine_Size']),
            'Engine_Cylinders': float(request.form['Engine_Cylinders']),
            'Drive_Type': request.form['Drive_Type'],
            'Fuel_Type': request.form['Fuel_Type'],
            'Vehicle Class/Type': request.form['Vehicle Class/Type'],
            'Car_Brand': request.form['Car_Brand'],
            'Model_Year': int(request.form['Model_Year']),
            'Fuel_Capacity': float(request.form['Fuel_Capacity'])
        }

        df_input = pd.DataFrame([input_data])
        one_hot_cols = ['Drive_Type', 'Fuel_Type', 'Vehicle Class/Type']
        df_input = pd.get_dummies(df_input, columns=one_hot_cols, drop_first=True)

        hash_enc = HashingEncoder(cols=['Car_Brand'], n_components=16)
        df_input = pd.concat([df_input.drop(columns=['Car_Brand']),
                              hash_enc.fit_transform(pd.DataFrame([{'Car_Brand': input_data['Car_Brand']}]))], axis=1)

        df_input = df_input.reindex(columns=columns, fill_value=0)
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)[0]
        interpretation = interpret_mpg(pred)

        return render_template_string(RESULT_TEMPLATE, mpg=f"{pred:.1f}", interpretation=interpretation)

    except Exception as e:
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Press Start 2P', cursive; background: #000; color: #ff0000; padding: 20px; }
                .container { max-width: 700px; margin: 0 auto; background: #1a1a1a; border: 4px solid #ff0000; padding: 40px; }
                h1 { color: #ff0000; text-shadow: 0 0 15px #ff0000; font-size: 1.5em; }
                p { color: #ffff00; font-size: 0.7em; margin: 20px 0; }
                .btn { background: #000; color: #ff0000; border: 3px solid #ff0000; padding: 15px 30px; font-family: 'Press Start 2P', cursive; cursor: pointer; font-size: 0.7em; }
                .btn:hover { background: #ff0000; color: #000; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⚠ GAME OVER ⚠</h1>
                <p>ERROR: """ + str(e) + """</p>
                <button class="btn" onclick="window.location.href='/'">★ RETRY ★</button>
            </div>
        </body>
        </html>
        """)

if __name__ == "__main__":
    app.run(debug=True)
