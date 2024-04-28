### BlockStar & AgentFormer


**BlockStar** - A multi-block A* algorithm that looks ahead to find how to rearrange the blocks (the base of a Genetic Algorithm)

**AgentFormer** - A character level transformer with a converter for the inputs and outputs from BlockStar. 


I worked as a labourer on a precast concrete site. We had to move lots of blocks around, and would often do way more work than necessary. 

![in action](https://raw.githubusercontent.com/rayking99/BlockStar/main/GA.gif)

If a number of blocks have to move to their target locations, in a specific order, other blocks may need to accomodate these moves - otherwise they may be in the way and prevent the blocks from arriving at their destination. 

The BlockStar algorithm was built for a genetic algorithm, where given n paths, the block would take a random one - fitness etc is calculated, there is  and passed through to the next generation.

On these random results the BlockStar achieves: 
Best: 86 Moves
Worst: 314 Moves (but also many failures)
Average: ~133 Moves. 

Results from successful BlockStar runs are saved to results.txt, fed into a transformer to train a language model, and decoded back into moves. 

Plug the moves back into the BlockStar at the end to evaluate. 

```

DIRECTIONS = ['up', 'down', 'left', 'right']
PIECES = ['A', 'C', 'E', 'F', 'G', 'K','B', 'L', 'P', 'Q', 'T', 'U', 'V','M','D','W']

state_1 = copy.deepcopy(start_state)
puzzle = Puzzle(state_1, target_state, PIECES)

moves = [['C', 'right', 1], ...['F', 'left', 1], ['F', 'up', 4]]

for move in moves:
    puzzle.apply_move(*move)
```

Inspired by 
- SearchFormer [(FacebookResearch) ](https://github.com/facebookresearch/searchformer)
- 'Let's build GPT: from scratch, in code, spelled out.' [(Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=88s)
