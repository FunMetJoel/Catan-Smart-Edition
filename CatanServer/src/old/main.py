from __future__ import annotations

import flask
from flask_cors import CORS
import old.catan as catan

app = flask.Flask(__name__)
CORS(app)
catanGame = catan.CatanState()

enforceRules = False

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

@app.route("/getRoads", methods=['GET'])
def get_roads():
    dataToSend = []
    for road in catanGame.board.roads:
        dataToSend.append(road.player)

    return flask.jsonify(dataToSend)

@app.route("/setRoad/<int:x>/<int:y>/<int:p>")
def set_road(x, y, p):
    catanGame.board.road(x, y).player = p
    return flask.jsonify("Road set")

@app.route("/getPossibleRoads/<int:p>")
def get_possible_roads(p):
    return flask.jsonify(catanGame.getRoadAvailabilty(p))

@app.route("/getSettlements", methods=['GET'])
def get_settlements():
    dataToSend = []
    for settlement in catanGame.board.settlements:
        dataToSend.append(settlement.player)
        dataToSend.append(settlement.level)

    return flask.jsonify(dataToSend)

@app.route("/setSettlement/<int:x>/<int:y>/<int:p>/<int:l>")
def set_settlement(x, y, p, l):
    if enforceRules:        
        if l == 1:
            if catanGame.players[p].hasResources([1, 1, 1, 1]) == False:
                return flask.jsonify("Not enough resources")
        
            possibleSettlements = catanGame.getSettlementAvailabilty(p)
            if possibleSettlements[catanGame.board.getsettlementindex(x, y)] == False:
                return flask.jsonify("Cannot build settlement here")
        
        if l == 2:
            if catanGame.players[p].hasResources([2, 3, 0, 0]) == False:
                return flask.jsonify("Not enough resources")
            
            if catanGame.board.settlement(x, y).level != 1:
                return flask.jsonify("Cannot build city here")
            
        
    catanGame.board.settlement(x, y).player = p
    catanGame.board.settlement(x, y).level = l
    return flask.jsonify("Settlement set")


@app.route("/getPossibleSettlements/<int:p>")
def get_possible_settlements(p):
    return flask.jsonify(catanGame.getSettlementAvailabilty(p))

@app.route("/getMaterials/<int:p>")
def get_materials(p):
    return flask.jsonify(catanGame.players[p].resources)

@app.route("/rollDice")
def roll_dice():
    return flask.jsonify(catanGame.rollDice())

if __name__ == '__main__':
        app.run(
        host='0.0.0.0',	
        port=5000,
        debug=False
    )