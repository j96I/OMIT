from flask import Flask, render_template, request

from omit import predict_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    if file:
        return predict_image(file)

if __name__ == '__main__':
    app.run(debug=True)
