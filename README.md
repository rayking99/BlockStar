### BlockStar & AgentFormer

I worked as a labourer on a precast concrete site. We had to move lots of blocks around, and would often do way more work than necessary. 

![in action](https://raw.githubusercontent.com/rayking99/BlockStar/main/GA.gif?token=GHSAT0AAAAAACK4JQVXIRNKKVF4AEAXBWLWZRNY5PA)

If a number of blocks have to move to their target locations, in a specific order, other blocks may need to accomodate these moves - otherwise they may be in the way and prevent the blocks from arriving at their destination. 

Results from successful BlockStar runs are saved to results.txt, fed into a transformer to train a language model, and decoded back into moves. 

Needs some fine-tuning, but the 

Inspired by 
- SearchFormer [(FacebookResearch) ](https://github.com/facebookresearch/searchformer)
- 'Let's build GPT: from scratch, in code, spelled out.' [(Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=88s)

**BlockStar** - A multi-block A* algorithm that looks ahead to find how to rearrange the blocks (hacky as)

**AgentFormer** - A character level transformer with a converter for the inputs and outputs from BlockStar. 

```

Output:

[['C', 'right', 1], ['C', 'down', 2], ['C', 'right', 2], ['Q', 'up', 6], ['G', 'left', 3], ['G', 'up', 9], ['W', 'down', 3], ['F', 'up', 8], ['M', 'right', 1], ['F', 'down', 2], ['P', 'left', 2], ['U', 'left', 2], ['W', 'down', 7], ['P', 'left', 1], ['G', 'right', 1], ['F', 'left', 5], ['Q', 'right', 10], ['U', 'right', 1], ['U', 'up', 4], ['G', 'left', 2], ['C', 'down', 6], ['M', 'right', 5], ['F', 'left', 4], ['C', 'down', 2], ['M', 'down', 1], ['C', 'up', 7], ['C', 'down', 8], ['C', 'right', 2], ['C', 'up', 3], ['C', 'right', 1], ['C', 'up', 6], ['C', 'left', 1], ['W', 'right', 2], ['W', 'up', 16], ['M', 'right', 9], ['M', 'up', 3], ['M', 'right', 1], ['M', 'up', 12], ['M', 'left', 2], ['M', 'up', 2], ['Q', 'down', 1], ['C', 'right', 2], ['U', 'down', 4], ['Q', 'right', 3], ['U', 'up', 1], ['Q', 'left', 2], ['C', 'right', 2], ['W', 'up', 4], ['W', 'right', 1], ['Q', 'down', 7], ['G', 'right', 2], ['M', 'up', 7], ['C', 'right', 2], ['C', 'right', 4], ['C', 'up', 8], ['C', 'left', 3], ['C', 'up', 7], ['C', 'left', 2], ['C', 'up', 1], ['C', 'left', 1], ['C', 'up', 4], ['C', 'right', 3], ['C', 'up', 7], ['C', 'left', 2], ['U', 'left', 5], ['U', 'down', 1], ['W', 'right', 3], ['W', 'up', 5], ['W', 'right', 1], ['W', 'up', 3], ['V', 'left', 3], ['V', 'down', 2], ['M', 'left', 3], ['M', 'up', 4], ['M', 'right', 1], ['M', 'up', 3], ['M', 'left', 2], ['G', 'down', 1], ['G', 'down', 2], ['G', 'right', 1], ['F', 'left', 3], ['F', 'left', 1], ['F', 'right', 10], ['F', 'up', 5], ['F', 'left', 2], ['F', 'up', 7], ['F', 'right', 2], ['F', 'up', 5], ['F', 'right', 2], ['F', 'up', 5], ['F', 'left', 2], ['U', 'right', 7], ['U', 'up', 14], ['G', 'right', 7], ['G', 'up', 8], ['G', 'right', 1], ['G', 'up', 8], ['G', 'right', 4], ['G', 'up', 3], ['P', 'right', 5], ['P', 'up', 7], ['T', 'right', 2], ['T', 'up', 3]]

len: 103

```
