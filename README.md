# Q-Learning-Gym-Cartpole

This repository implements Tabular Q-learning for the cart pole task using OpenAI Gym. Gym module provides an abstraction for Markov decision processes, enabling the rapid development and test of reinforcement learning algorithms. Instructions about Cartpole Gym can be found under https://github.com/openai/gym.

The convergence of the algorithm depends on the hyperparameter settings. Due to my implementation the hyperparameters are the following:
<ul>
  <li>The number of bins. I used Binning as quantization method of the continuous state values. With a very fine-grained discrete state representation, there are lots of values to fill in the table. With a very coarsegrained number of bins, the Q table might not have enough complexity to solve the task.</li>
  <li>Learning rate (step size), α. Too much, the algorithm diverge. Too small, it will need a lot of steps (and possibly a lot of exploration)</li>
  <li>Discounting factor, γ. 0 would mean no horizon, will learn faster, but short-sighted. 1: full horizon, will
learn slower, though the agent can see long horizons.</li>
  <li>ϵ factor for exploration. 0: full exploitation, 1: full exploration.</li>
</ul>

My choice of hyperparameters allowed to see a convergence with a relatively small number of iterations (600).
The latest implementation can be found in the Jupyter Notebook Gym Cartpole Q-Learning.ipynb.

