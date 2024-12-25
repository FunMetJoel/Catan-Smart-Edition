import flask
from flask_cors import CORS

import catanData as Catan
import compiledCordinateSystem as ccs

from firstNN import firstNN
import numpy as np
import random
from compiledCordinateSystem import compiledCornerIndex

app = flask.Flask(__name__)
CORS(app)
ai = firstNN()

catanGame = Catan.CatanState()

for i in range(4):
    for j in range(2):
        cornerMask = catanGame.getCornerMask(i)
        randomIndex = np.random.randint(52)
        while not cornerMask[randomIndex]:
            randomIndex = np.random.randint(52)
        catanGame.corners[randomIndex][0] = i+1
        catanGame.corners[randomIndex][1] = 1
        
        edgeMask = catanGame.getEdgeMask(i)
        ramdomRoadIndex = random.choice(compiledCornerIndex.neighbourEdges[randomIndex])
        while not edgeMask[ramdomRoadIndex]:
            ramdomRoadIndex = random.choice(compiledCornerIndex.neighbourEdges[randomIndex])
        catanGame.edges[ramdomRoadIndex][0] = i+1
        catanGame.edges[ramdomRoadIndex][1] = 1
        
catanGame.round = 2

enforceRules = False

users = []

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


def RUN():
    app.run(
        host='0.0.0.0',	
        port=5000,
        debug=False
    )
    
if __name__ == '__main__':
    RUN()
