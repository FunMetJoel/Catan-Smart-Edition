# Server for catan game
# Should handle:
# - Connecting to the game
# - Making moves
# - Getting game state
# - Resetting game
# - Vieuwer and player

import flask
from flask_cors import CORS
import datetime as dt
import time
import threading
import logging

import catanData as Catan
import compiledCordinateSystem as ccs
import firstTree as firstTreeBot



catanGame = Catan.CatanState()

players = [
    ('bot', firstTreeBot.firstTreeBot()),
    ('bot', firstTreeBot.firstTreeBot()),
    ('bot', firstTreeBot.firstTreeBot()),
    ('bot', firstTreeBot.firstTreeBot()),
    # ('player', '127.0.0.1'),
    # ('player', '127.0.0.1'),
    # ('player', '127.0.0.1'),
    # ('player', '127.0.0.1')
]
CardsOpen = False

lastMoveTimestamp = 0
gameStartTimestamp = 0

tree = firstTreeBot.firstTreeBot()

def mainloop():
    global catanGame
    global players
    global lastMoveTimestamp
    global gameStartTimestamp
    setupBoard()
    lastPlayer = -1
    while True:
        if lastPlayer != catanGame.currentPlayer:
            print(catanGame.currentPlayer, )
            lastPlayer = catanGame.currentPlayer
        currentPlayer = players[catanGame.currentPlayer]
        if currentPlayer[0] == 'bot':
            currentPlayer[1].make_move(catanGame)
            lastMoveTimestamp = dt.datetime.now()
        else:
            pass
        
        time.sleep(0.25)
        
        if catanGame.points.max() >= 10:
            setupBoard()
            
        if (dt.datetime.now() - lastMoveTimestamp).seconds > 60:
            catanGame.endTurn()
            lastMoveTimestamp = dt.datetime.now()
            print(" Turn ended because of timeout")
             
def setupBoard():
    global catanGame
    global lastMoveTimestamp
    global gameStartTimestamp
    catanGame = Catan.CatanState()
    
    while catanGame.round < 2:
        tree.make_opening_move(catanGame)
        catanGame.endTurn()
        
    lastMoveTimestamp = dt.datetime.now()
    gameStartTimestamp = dt.datetime.now()
    
    
app = flask.Flask(__name__)
CORS(app)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

##########
# checks #
##########

def checkIfAllowedRoute(player, ip, isGet):
    global players
    if isGet:
        if CardsOpen:
            return True
    else:
        if player != catanGame.currentPlayer:
            return False
        
    if players[player][0] == 'bot':
        return False
    if players[player][1] == ip:
        return True
    
    return False
    
    
###################
# algemene routes #
###################

@app.route('/ping')
def ping():
    return flask.jsonify("pong")

@app.route('/getIp')
def get_ip():
    return flask.jsonify(flask.request.remote_addr)

@app.route('/getGameStartTimestamp')
def get_game_start_timestamp():
    return flask.jsonify(gameStartTimestamp.timestamp())

#################
# viewer routes #
#################

@app.route('/getCurrentPlayer')
def get_current_player():
    return flask.jsonify(catanGame.currentPlayer)

@app.route('/getPoints/<int:p>')
def get_points(p):
    return flask.jsonify(catanGame.points[p-1].item())

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
    # print(dataToSend)
    return flask.jsonify(dataToSend)

@app.route('/getLastRoll')
def get_last_roll():
    return flask.jsonify(catanGame.lastRoll)




@app.route('/makeOpeningMove')
def make_opening_move():
    tree.make_opening_move(catanGame)
    catanGame.endTurn()
    
    return flask.jsonify(f"Opening Move Made")



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

@app.route('/setSettlement/<int:x>/<int:y>')
def set_settlement(x, y):
    if not checkIfAllowedRoute(catanGame.currentPlayer, flask.request.remote_addr, False):
        return flask.jsonify("Not Allowed")
    index = ccs.CompiledCornerIndex.calcCornerIndex(x, y)
    catanGame.buildSettlement(index)
    return flask.jsonify("Settlement set")

@app.route('/setCity/<int:x>/<int:y>')
def set_city(x, y):
    if not checkIfAllowedRoute(catanGame.currentPlayer, flask.request.remote_addr, False):
        return flask.jsonify("Not Allowed")
    index = ccs.CompiledCornerIndex.calcCornerIndex(x, y)
    catanGame.buildCity(index)
    return flask.jsonify("City set")

@app.route('/setRoad/<int:x>/<int:y>')
def set_road(x, y):
    if not checkIfAllowedRoute(catanGame.currentPlayer, flask.request.remote_addr, False):
        return flask.jsonify("Not Allowed")
    index = ccs.CompiledEdgeIndex.calcEdgeIndex(x, y)
    catanGame.buildRoad(index)
    return flask.jsonify("Road set")

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



@app.route('/trade/<int:give>/<int:get>')
def trade(give, get):
    catanGame.tradeWithBank(give, get)
    return flask.jsonify("Trade Made")

@app.route('/endTurn')
def end_turn():
    catanGame.endTurn()
    return flask.jsonify("Turn Ended")


def RUN():
    threading.Thread(target=mainloop).start()
    app.run(
        host='0.0.0.0',	
        port=5000,
        debug=False
    )
    
if __name__ == '__main__':
    RUN()
            
        
        