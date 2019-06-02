# 1. Learning Algorithm

## Agent and DDPG Algorithm
The agent uses deterministic deep policy gradient (DDPG) as described in [this](https://arxiv.org/abs/1509.02971) article. It has one critic network evaluating the current state and action and an actor network returning  continous actions to a given (continous) state. For both, critic and actor, local and networks are used.

The principal algorithm as given in the [paper](https://arxiv.org/abs/1509.02971) is:
* **Initialize** local and target critic network Q, Q' and local and target actor network μ, μ', replay buffer R
* **Loop** over episodes
* --**Initialize** a random process N for exploration
* --**Receive** initial observation state s
* --**Loop** over time steps 
* ----**Select action** a_t = μ(s_t | θ_μ) + N_t according to policy and exploration noise
* ----**Execute** action and observe reward r_t and new state s_t+1
* ----**Store** transition (s_t, a_t, r_t, s_t+1) in R
* ----**Sample** minibatch of n transitions (s_i, a_i, r_i, s_i+1) from R
* ----**Set** y_i = r_i + γQ'(s_i+1, μ'(s_i+1 | θ_μ') | θ_Q')
* ----**Update Critic** by minimizing loss L = 1/n Σ_i (y_i - Q(s_i, a_i | θ_Q))^2 with respect to θ_Q
* ----**Update Policy** by minimizing actor loss J = -Q(s_i, μ(s_i | θ_μ)) with respect to θ_μ
* ----**Soft transition** from local to target networks Q --> Q', μ --> μ'
* --**End Loop**
* **End Loop**

Where these steps are repeated for each step in each episode. Note that the epsilon value for choosing the action epsilon greedy is reduced linearly over episodes, starting with `eps_start = 1.0` and ending at `eps_end = 0.01` after `eps_nEpisodes = 1000`. 

Further agent hyperparameters are:

	BUFFER_SIZE = int(1e5)  # replay buffer size
	BATCH_SIZE = 64         # minibatch size for learning
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR = 5e-4               # learning rate 
	UPDATE_EVERY = 4        # how often to update the network
	
As optimizer `Adam` has been used.

## Neural Network
As the observation space of the environment is `state_size = 37` the input size of the neural network matches this size and as `action_size = 4` the output size of the neural network matches this as well. Between input and output are two linear hidden layers, both with size `hidden_layers = [37*3, 37*3]` and `relu` activation. 

# 2. Plot of Rewards
With the above described agent the environment has been solved in 1029 episodes. The development of average rewards as well with all scores over each episode are provided below.

	Episode 100	Average Score: 0.31
	Episode 200	Average Score: 2.32
	Episode 300	Average Score: 4.46
	Episode 400	Average Score: 7.93
	Episode 500	Average Score: 11.21
	Episode 551	Average Score: 13.00
	Environment solved in 551 episodes!	Average Score: 13.00

![Score over Episodes for DQN agent](./data/score_over_episodes_report.png "Score over Episodes")

# 3. Ideas for Future Work
As the DQN algorithm overestimates the action values by taking always the maximum of noisy data, the algorithm can be extended to a Double DQN (DDQN). In DDQN when evaluating the best action value the process is splitted into selection of the best action and evaluation of the best action. So both have to agree on the best action, otherewise the resulting action value is not as high. For the evaluation part the already implemented target neural network can be reused. This [paper](https://arxiv.org/abs/1509.06461) gives the details about this concept and how DQN overestimates the action values.

Another thing worth investigating further is prioritized experience replay. The main idea behind this concept is, that not all experiences are equally important. Therefore to each experience / transition the TD-error is added as a measure of priority. With this a probability can be formulated, favoring experiences more important. Though some further extensions are necessary to assure a certain amount of randomness and the update rule has to incorporate this changed selection process as well. More details can be found in this [paper](https://arxiv.org/abs/1511.05952).