import argparse
import os
import logging

from flask import Flask
from flask import request, jsonify, render_template

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["gpu"] = None
app.config["SERVER_NAME"] = "localhost:6800"


def get_music_styles():
    return {'rock_style': 'Rock', 'rap_style': 'Rap', 'pop_style': 'Pop'}

@app.route('/')
@app.route('/index')
def main():
    return render_template('main.html', music_styles=get_music_styles())


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_data()
    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_false')
    # TO-DO
    args = parser.parse_args()
    app.run(debug=args.debug)