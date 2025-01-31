import flask
from flask_cors import CORS

import catanData as Catan
import compiledCordinateSystem as ccs

from fourthNN import forthNN
from firstTree import firstTreeBot
import numpy as np
import random
from compiledCordinateSystem import compiledCornerIndex
import torch

app = flask.Flask(__name__)
CORS(app)
tree = firstTreeBot()
ai = forthNN()
ai.model.load_state_dict(torch.load("src/bots/2025-01-10-22-41-47/model4NN-1.549-2025-01-10-22-41-47.pth"))

catanGame = Catan.CatanState()

while catanGame.round < 2:
    tree.make_opening_move(catanGame)
    catanGame.endTurn()
        
enforceRules = False

users = []

@app.route('/ping')
def ping():
    return flask.jsonify("pong")

@app.route('/makeOpeningMove')
def make_opening_move():
    tree.make_opening_move(catanGame)
    catanGame.endTurn()
    
    return flask.jsonify(f"Opening Move Made")

@app.route('/getCurrentPlayer')
def get_current_player():
    return flask.jsonify(catanGame.currentPlayer)

@app.route('/resetGameIfOver')
def reset_game_if_over():
    global catanGame
    if catanGame.points.max() >= 10:
        catanGame = Catan.CatanState()
        
        while catanGame.round < 2:
            tree.make_opening_move(catanGame)
            catanGame.endTurn()
        
        return flask.jsonify("Game Reset")
    else:
        return flask.jsonify("Game Not Over")

@app.route('/resetGame')
def reset_game():
    global catanGame
    catanGame = Catan.CatanState()
    
    while catanGame.round < 2:
        tree.make_opening_move(catanGame)
        catanGame.endTurn()
        
    return flask.jsonify("Game Reset")

@app.route('/getAvailableActions')
def get_available_actions():
    actions = catanGame.getActionMask().tolist()
    return flask.jsonify(actions)

@app.route('/calculateLongestRoad')
def calculate_longest_road():
    lengts = []
    for i in range(4):
        lengts.append(catanGame.calculateLongestRoad(i))
    return flask.jsonify(lengts)

@app.route('/')
def home():
    return "Welcome to Catan Smart Edition!"

@app.route('/getPossibleRoads/<int:p>')
def get_possible_roads(p):
    return flask.jsonify(catanGame.getEdgeMask(p-1).tolist())

@app.route('/getPossibleSettlements/<int:p>')
def get_possible_settlements(p):
    return flask.jsonify(catanGame.getCornerMask(p-1).tolist())

@app.route('/setSettlement/<int:x>/<int:y>/<int:p>/<int:l>')
def set_settlement(x, y, p, l):
    index = ccs.CompiledCornerIndex.calcCornerIndex(x, y)
    catanGame.buildSettlement(index)
    return flask.jsonify("Settlement set")

@app.route('/setCity/<int:x>/<int:y>/<int:p>')
def set_city(x, y, p):
    index = ccs.CompiledCornerIndex.calcCornerIndex(x, y)
    catanGame.buildCity(index)
    return flask.jsonify("City set")

@app.route('/setRoad/<int:x>/<int:y>/<int:p>')
def set_road(x, y, p):
    index = ccs.CompiledEdgeIndex.calcEdgeIndex(x, y)
    catanGame.buildRoad(index)
    return flask.jsonify("Road set")

@app.route('/getSettlements')
def get_settlements():
    dataToSend = catanGame.corners.flatten().tolist()

    return flask.jsonify(dataToSend)

@app.route('/getRoads')
def get_roads():
    dataToSend = []
    for road in catanGame.edges:
        dataToSend.append(road[0].item())
    return flask.jsonify(dataToSend)

@app.route('/getTile/<int:x>/<int:y>')
def get_tile(x, y):
    dataToSend = {
        "tile_type": catanGame.hexes[ccs.CompiledHexIndex.gethexindex(x, y)][0].item(),
        "number": catanGame.hexes[ccs.CompiledHexIndex.gethexindex(x, y)][1].item()
    }
    print(dataToSend)
    return flask.jsonify(dataToSend)

@app.route('/makeAIMove')
def make_ai_move():
    ai.make_move(catanGame)
    return flask.jsonify("AI Move Made")

@app.route('/playAiGame')
def play_ai_game():
    for i in range(100):
        ai.make_move(catanGame)
    return flask.jsonify("Ai Made 100 Moves")

@app.route('/playUntilPlayer/<int:p>')
def play_until_player(p):
    while catanGame.currentPlayer != p:
        ai.make_move(catanGame)
    return flask.jsonify("Ai Made Moves Until Player "+str(p))

@app.route('/getMaterials/<int:p>')
def get_materials(p):
    return flask.jsonify(catanGame.players[p-1].tolist())

@app.route('/getPoints/<int:p>')
def get_points(p):
    return flask.jsonify(catanGame.points[p-1].item())

@app.route('/trade/<int:give>/<int:get>')
def trade(give, get):
    catanGame.tradeWithBank(give, get)
    return flask.jsonify("Trade Made")

@app.route('/endTurn')
def end_turn():
    catanGame.endTurn()
    return flask.jsonify("Turn Ended")


def RUN():
    app.run(
        host='0.0.0.0',	
        port=5000,
        debug=False
    )
    
if __name__ == '__main__':
    RUN()

