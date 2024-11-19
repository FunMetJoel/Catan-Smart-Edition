import flask
import catan

app = flask.Flask(__name__)
catanGame = catan.CatanState()

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

# Get tile at x, y
@app.route("/getTile/<int:x>/<int:y>", methods=['GET'])
def get_tile(x, y):
    tile = catanGame.board.hex(x, y)
    dict = tile.to_dict()
    print(dict)
    print(type(dict))
    return flask.jsonify(tile.to_dict())


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',	
        port=5000,
        debug=True
    )