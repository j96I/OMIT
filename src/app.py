from flask import Flask, render_template, request

from omit import predict_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = file.filename
        print(file.content_type)
        print(filename)
        return predict_image(file)

if __name__ == '__main__':
    app.run(debug=True)
