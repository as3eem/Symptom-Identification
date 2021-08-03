from flask import Flask, request, render_template, jsonify
from model import predict_symptom

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', size=0)


@app.route('/', methods=['POST'])
def get_prediction():
    title = request.form.get('symptoms')
    print()
    out = predict_symptom(title)
    print(out)
    
    ## give mappings returned
    result=out
    # if out:
    #     result = 'SPAM'
    # else:
    #     result = 'Not-SPAM'

    print(result, len(result),'+++++++++')
    return render_template('index.html', size=len(out), symptoms=result)


if __name__ == '__main__':
    app.run(port='5000', debug=True)