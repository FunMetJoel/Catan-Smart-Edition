import torch
import torch.nn as nn
import numpy as np
import old.catan as catan
import old.botobject as botobject
import cordinatesystem
import enums

class FirstAI(botobject.CatanBot):
    def __init__(self):
        super().__init__("FirstAI", "A simple AI that makes random moves")
        self.model = None

    def load_model(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        
    def save_model(self, model_path):
        torch.save(self.model, model_path)
        
    def new_model(self):
        self.model = CatanAiModel()
        
    def interpret_player(self, dataPlayer:int, currentPlayer:int):
        # input: 0 for no player, 1 for player 1, 2 for player 2, 3 for player 3, 4 for player 4
        # output: 0 for no player, 1 for current player, -1 for other player
        if dataPlayer == 0:
            return 0
        elif dataPlayer == currentPlayer:
            return 1
        else:
            return -1
        
    def interpret_state(self, game_state: catan.CatanState):
        # 19 tensors of 62 values each
        current_player = game_state.current_player
        hextensors = [torch.zeros(21)] * 19
        for i, hex in enumerate(game_state.board.tiles):
            corners:list[catan.CatanSettlement] = []
            for corner in enums.HexCords[i].corners():
                corners.append(game_state.board.settlement(corner.x, corner.y))
                
            edges:list[catan.CatanRoad] = []
            for edge in enums.HexCords[i].edges():
                edges.append(game_state.board.road(edge.x, edge.y))
                
            hexdata = [
                hex.tile_type,
                hex.number,
                hex.robber,
                self.interpret_player(corners[0].player, current_player),
                corners[0].level,
                self.interpret_player(corners[1].player, current_player),
                corners[1].level,
                self.interpret_player(corners[2].player, current_player),
                corners[2].level,
                self.interpret_player(corners[3].player, current_player),
                corners[3].level,
                self.interpret_player(corners[4].player, current_player),
                corners[4].level,
                self.interpret_player(corners[5].player, current_player),
                corners[5].level,
                self.interpret_player(edges[0].player, current_player),
                self.interpret_player(edges[1].player, current_player),
                self.interpret_player(edges[2].player, current_player),
                self.interpret_player(edges[3].player, current_player),
                self.interpret_player(edges[4].player, current_player),
                self.interpret_player(edges[5].player, current_player)
            ]
            
            for j in range(21):
                hextensors[i][j] = float(hexdata[j])

    
        playertensors = [torch.zeros(6)] * 4
        
        for i, player in enumerate(game_state.players):
            playerdata = [
                player.resources[0],
                player.resources[1],
                player.resources[2],
                player.resources[3],
                player.resources[4],
                0
            ]
            
            for corner in game_state.board.settlements:
                if corner.player == i:
                    playerdata[5] += corner.level
                    
            for j in range(6):
                playertensors[i][j] = float(playerdata[j])
                    
        return hextensors, playertensors
            
            

    def make_move(self, game_state):
        hexes, players = self.interpret_state(game_state)
        
        with torch.no_grad():
            actions, roads, settlements = self.model(hexes, players)
            
        actions = actions.numpy()
        roads = roads.numpy()
        settlements = settlements.numpy()
        
        actionMask = game_state.getActionAvailability()
        roadMask = game_state.getRoadAvailabilty()
        settlementMask = game_state.getSettlementAvailabilty()
        
        actions = actions * actionMask
        roads = roads * roadMask
        settlements = settlements * settlementMask
        print(actions)
        # print(roads)
        # print(settlements)
        
        action = np.argmax(actions)
        print(action)
        if action == 0:
            game_state.endTurn()
        elif action == 1:
            settlement = np.argmax(settlements)
            game_state.board.settlements[settlement].player = game_state.current_player
            game_state.board.settlements[settlement].level = 1
            game_state.players[game_state.current_player-1].resources[0] -= 1
            game_state.players[game_state.current_player-1].resources[1] -= 1
            game_state.players[game_state.current_player-1].resources[2] -= 1
            game_state.players[game_state.current_player-1].resources[3] -= 1
        elif action == 2:
            city = np.argmax(settlements)
            game_state.board.settlement(city).player = game_state.current_player
            game_state.board.settlement(city).level = 2
            game_state.players[game_state.current_player-1].resources[0] -= 2
            game_state.players[game_state.current_player-1].resources[1] -= 3
        elif action == 3:
            road = np.argmax(roads)
            game_state.board.roads[road].player = game_state.current_player
            game_state.players[game_state.current_player-1].resources[1] -= 1
            game_state.players[game_state.current_player-1].resources[3] -= 1
            

class CatanAiModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 19 hexes, 21 values each and 4 players, 6 values each
        # Define the heads for the 21-size input vectors
        self.hexes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(21, 64),
                nn.ReLU()
            ) for _ in range(19)
        ])

        # Define the heads for the 6-size input vectors
        self.players = nn.ModuleList([
            nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU()
            ) for _ in range(4)
        ])

        # Example output processing layer, combining all head outputs
        self.fullyConnected1 = nn.Sequential(
            nn.Linear(19 * 64 + 4 * 32, 128*32),
            nn.ReLU()
        )
        
        self.fullyConnected2 = nn.Sequential(
            nn.Linear(128*32, 64*32),
            nn.ReLU()
        )
        
        self.actionHead = nn.Sequential(
            nn.Linear(64*32, 64*32),
            nn.ReLU()
        )
        
        self.roadsHead = nn.Sequential(
            nn.Linear(64*32, 64*32),
            nn.ReLU()
        )
        
        self.settlementsHead = nn.Sequential(
            nn.Linear(64*32, 64*32),
            nn.ReLU()
        )
        
        self.outputAction = nn.Sequential(
            # 1000 = end turn, 0100 = build settlement, 0010 = build city, 0001 = build road
            nn.Linear(64*32, 4),
            nn.Sigmoid()
        )
        
        self.outputRoads = nn.Sequential(
            nn.Linear(64*32, 72),
            nn.Sigmoid()
        )
        
        self.outputSettlements = nn.Sequential(
            nn.Linear(64*32, 54),
            nn.Sigmoid()
        )
        
    def forward(self, input_hexes, input_players):
        # Apply each head to the corresponding vector in input_21
        processed_hexes = [head(vector) for head, vector in zip(self.hexes, input_hexes)]
        processed_hexes = torch.cat(processed_hexes, dim=-1)  # Concatenate along the feature dimension

        # Apply each head to the corresponding vector in input_6
        processed_players = [head(vector) for head, vector in zip(self.players, input_players)]
        processed_players = torch.cat(processed_players, dim=-1)  # Concatenate along the feature dimension

        # Combine all processed outputs
        combined = torch.cat([processed_hexes, processed_players], dim=-1)

        # Final output processing
        w = self.fullyConnected1(combined)
        w = self.fullyConnected2(w)
        x = self.actionHead(w)
        y = self.roadsHead(w)
        z = self.settlementsHead(w)
        
        action = self.outputAction(x)
        roads = self.outputRoads(y)
        settlements = self.outputSettlements(z)
        
        return action, roads, settlements
        
if __name__ == "__main__":
    print("This is a module for the FirstAI bot")
    print("Please run src/main.py to start the game")
    print("--------------------------------------")
    Input = input("Do you want to train the model? (y/n): ")
    if Input == "y":
        print("Training the model...")
        models = [ FirstAI() for _ in range(4) ]
        for model in models:
            model.new_model()
        
        gamestate = catan.CatanState()
        
        for i in range(4):
            #chose random place to set a settlement
            settlement = np.random.randint(0, 54)
            gamestate.board.settlements[settlement].player = i
            gamestate.board.settlements[settlement].level = 1
        
        gamestate.round = 2
        print(gamestate.round)
        
        while not gamestate.isFinished():
            currentPlayer = gamestate.current_player
            gamestate.nextPlayer()
            print('Player', currentPlayer, ' turn')
            #models[currentPlayer-1].make_move(gamestate)
            #print('Player', currentPlayer, ' made a move: ', )
        
        
    else:
        print("Model not trained")