from flask import Flask, render_template, request

from omit import predict_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    # r1 - asyncio.to_thread(requests.get, 'API endpoint')

    # results - await [r1, r2]

    # call api - response in the form of  {tags: ['asdf', 'zxcv]}

    # for each item in response array massage into list - format of <li>data</li>

    # concat outputs - format: <li>tag1</li><li>tag2</li>

    # wrap with ul - format: <ul><li>tag1</li><li>tag2</li></ul>


    if file:
        data = [predict_image(file), 'item2', 'item3']
        list_items = ''.join(f'<a>{item}</a>' for item in data)
        return list_items

if __name__ == '__main__':
    app.run(debug=True)
