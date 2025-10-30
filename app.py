from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Raw')
def kicaw():
    return "<h1>Hello mate!</h1>"


if __name__ == "__main__":
    app.run(debug=True, port=2611)