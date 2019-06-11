# 1. Learning Algorithm

## Agent and DDPG Algorithm
The agent uses deterministic deep policy gradient (DDPG) as described in [this](https://arxiv.org/abs/1509.02971) article. It has one critic network evaluating the current state and action and an actor network returning  continous actions to a given (continous) state. For both, critic and actor, local and target networks are used.

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

Where these steps are repeated for each step in each episode. Note that the epsilon value for choosing the action epsilon greedy is reduced linearly over episodes, starting with `eps_start = 1.5` and ending at `eps_end = 0.01` after `eps_nEpisodes = 2500`. 

Further agent hyperparameters are:

	BUFFER_SIZE = int(5e5)  # replay buffer size
	BATCH_SIZE = 64         # minibatch size for learning
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR_CRIT = 1e-3          # learning rate critic
	LR_ACTR = 1e-4          # learning rate actor
	WEIGHT_DECAY = 1e-2     # L2 weight decay
	UPDATE_EVERY = 1        # how often to update the network
	GD_EPOCH = 1            # how often to optimize when learning is triggered
	CLIP_GRAD = 1           # clipping gradient with this value
		
As optimizer `Adam` has been used.

## Neural Network
As the observation space of the environment is `state_size = 33` and `action_size = 4`, the input size of the critic network Q(s_i, a_i | θ_Q) is 37 and output size is 1. The input size of the actor μ(s_i | θ_μ) is equal to the state size 33. For both networks two linear hidden layers are used, with size `hidden_layers = [37, 37]` for the critic and `hidden_layers = [33, 33]` for the actor, both with `relu` activation. For the actor the output function `tanh` is used to keep outputs between -1 and 1.

# 2. Plot of Rewards
With the above described agent the environment has been solved in 1858 episodes. The development of average rewards as well with all scores over each episode are provided below.

![Score over Episodes for DDPG agent](./data/DDPG_results_report.png "Score over Episodes")

The above result shows a succesful and robust way of learning but takes a lot of time. Faster results can be achieved, but this is not really stable as can be seen in the next plot with 3 simulations where exploration is reduced from `eps_start = 1.0` to `eps_end = 0.01` after `eps_nEpisodes = 1800` and gradient clipping is deactivated. While the first run was able to reach the goal of 30 quite fast the following two runs were not able to reach the target.

![Compare Scores over Episodes for DDPG agent](./data/DDPG_compare_runs.png "Compare runs")


# 4. Ideas for Future Work
Training of the DDPG algorithm takes a lot of time and a lot of exploration is needed in order to get stable results. Therefore other algorithms maybe worth investigating such as Trust Region Policy Optimization (TRPO), which seems to provide a fast and stable learning according to [this paper](https://arxiv.org/pdf/1604.06778.pdf) or PPO which ist close to TRPO but seems to be a little easier to implement an can be found [here](https://arxiv.org/pdf/1707.06347.pdf).