import argparse

from flask import Flask
from flask import request, jsonify, render_template

from models.lyrics_classifier import LyricsClassifier

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["gpu"] = None
app.config["SERVER_NAME"] = "localhost:6800"


def get_music_styles():
    return {'sgdclassifier': 'SGDClassifier'}


@app.route('/')
@app.route('/index')
def main():
    return render_template('main.html', music_styles=get_music_styles())


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    model = LyricsClassifier.build()
    prediction = model.predict([data['sentence']]).tolist()
    print(prediction)
    return jsonify({"genre": prediction})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    app.run(debug=args.debug)
