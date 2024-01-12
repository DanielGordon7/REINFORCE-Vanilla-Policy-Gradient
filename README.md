Package requirements: pip install gymnasium

Vanilla Policy Gradient algorithm with reward-to-go undiscounted.

The reward-to-go makes the sample estimates have lower variance.
This ensures replicability of the code.
By using entire episode return there's lots of variance in performance.

Potential improvements:
1) Discount rewards by implementing recursive relation in same loop.

2) Use average of several sample estimates to decrease variance.

3) Implement baseline to further reduce variance.
This can include another Neural Network to compute Value Function.
This is close to the Actor-Critic framework for RL algorithms.
