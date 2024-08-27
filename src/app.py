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
        tag_list = [predict_image(file), 'item2', 'item3']
        return ''.join(f'<a>{tag}</a>' for tag in tag_list)

if __name__ == '__main__':
    app.run(debug=True)