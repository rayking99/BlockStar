### BlockStar & AgentFormer

I worked as a labourer on a precast concrete site. We had to move lots of blocks around, and would often do way more work than necessary. 

![in action]([https://raw.githubusercontent.com/rayking99/BlockStar/main/GA.gif?token=GHSAT0AAAAAACK4JQVXIRNKKVF4AEAXBWLWZRNY5PA](https://raw.githubusercontent.com/rayking99/BlockStar/main/GA.gif))

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

[['C', 'right', 1], ['C', 'down', 2], ['C', 'right', 2], ['Q', 'up', 6], ['G', 'left', 3] ... ['T', 'up', 3]]

```
