import flask

app = flask.Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Catan Smart Edition!"

@app.route('/api/v1/players', methods=['GET'])
def get_players():
    return "List of players"

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',	
        port=5000,
        debug=True
    )