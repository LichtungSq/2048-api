import json
import numpy as np
from game2048.game import Game
from game2048.displays import Display


def generate_fingerprint(AgentClass, **kwargs):
    with open("board_cases.json") as f:
        board_json = json.load(f)

    game = Game(size=4, enable_rewrite_board=True)
    agent = AgentClass(game=game, **kwargs)

    trace = []
    for board in board_json:
        game.board = np.array(board)
        direction = agent.step()
        trace.append(direction)
    fingerprint = "".join(str(i) for i in trace)
    return fingerprint


if __name__ == '__main__':
    from collections import Counter

    '''====================
    Use your own agent here.'''
    from game2048.agents import ExpectiMaxAgent
    game = Game(4, score_to_win=2048, random=True)
    display2 = Display()
    display2.display(game)
    TestAgent = ExpectiMaxAgent(game, display=display2)
    TestAgent.play(verbose=True)
    '''===================='''
    
    fingerprint = generate_fingerprint(TestAgent)

    with open("EE369_fingerprint.json", 'w') as f:        
        pack = dict()
        pack['fingerprint'] = fingerprint
        pack['statstics'] = dict(Counter(fingerprint))
        f.write(json.dumps(pack, indent=4))
