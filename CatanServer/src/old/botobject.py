import old.catan as catan

class CatanBot():
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def make_move(self, game_state: catan.CatanState) -> None:
        pass