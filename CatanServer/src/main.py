import flask
from flask_cors import CORS
import catan

app = flask.Flask(__name__)
CORS(app)
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

@app.route('/ping', methods=['GET'])
def connect():
    return flask.jsonify("pong")

# Get tile at x, y
@app.route("/getTile/<int:x>/<int:y>", methods=['GET'])
def get_tile(x, y):
    tile = catanGame.board.hex(x, y)    
    return flask.jsonify(tile.to_dict())

if __name__ == '__main__':
        app.run(
        host='0.0.0.0',	
        port=5000,
        debug=True
    )