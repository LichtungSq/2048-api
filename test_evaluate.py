from game2048.game import Game
from game2048.displays import Display
from game2048.agents import MyOwnAgent as TestAgent

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display = Display(), **kwargs)
    agent.play(verbose=True)
    return game.score

if __name__ == '__main__':
    
    count = 0
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    
    while True: 
        count += 1  
        score = single_run(GAME_SIZE, SCORE_TO_WIN, AgentClass=TestAgent)
        print("Scores: @%d time" % count, score)
        
