import catanData

class CatanBot():
    def make_move(self, game_state: catanData.CatanState):
        raise NotImplementedError("make_move not implemented")
    
    def make_opening_move(self, game_state: catanData.CatanState):
        '''This function is called in the first 2 rounds of the game.
        '''
        raise NotImplementedError("make_opening_move not implemented")
    
    