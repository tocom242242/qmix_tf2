# QMIX implemented in TensorFlow 2

Note: Currently, we are only experimenting with Two State Game.

## How to run QMIX in Two State Game

First, install packages.

```
pip install -r requirements.txt
```

Run main.py

```
python main.py
```

Then, outputs episode reward and qmix's loss history graph.

![result](https://github.com/tocom242242/qmix_tf2/blob/master/reward_loss_history.png)


## Reference
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
