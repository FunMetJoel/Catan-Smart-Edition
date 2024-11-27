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

@app.route("/getRoads")
def get_roads():
    dataToSend = []
    for road in catanGame.board.roads:
        dataToSend.append(road.player)

    return flask.jsonify(dataToSend)

@app.route("/setRoad")
def set_road(x, y, p):
    catanGame.board.road(x, y).player = p

if __name__ == '__main__':
        app.run(
        host='0.0.0.0',	
        port=5000,
        debug=True
    )