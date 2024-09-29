from flask import Flask, render_template, request
from omit import predict_image
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health.html')
def health():
    return 'healthy'


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    # fire_url = 'http://10.0.40.136:5000/tags/random'
    # res = requests.get(fire_url).json()
    # data.extend(res)

    if file:
        return f'<a>{predict_image(file)}</a>'
    
        # tag_list = [predict_image(file), 'item2', 'item3']
        # return ''.join(f'<a>{tag}</a>' for tag in tag_list)

        

if __name__ == '__main__':
    app.run(debug=True)