import flask

app = flask.Flask(__name__)

users = []

@app.route('/')
def home():
    return "Welcome to Catan Smart Edition!"

@app.route('/api/v1/players', methods=['GET'])
def get_players():
    global users
    ip = flask.request.remote_addr
    users.append(ip)
    return flask.jsonify(users)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',	
        port=5000,
        debug=True
    )