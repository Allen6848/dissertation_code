class TD3Agent:
    def __init__(self, state_size, discrete_size, gamma=0.99, learning_rate=1e-4, max_action=20):
        self.state_size = state_size
        self.discrete_size = discrete_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_action = max_action
        self.action_counter = 0
        self.model = ActorCritic(state_size)  # Action space size equals the number of discrete actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state, epsilon=0.1):
        """
        Get action using the actor-critic model with epsilon-greedy exploration.

        Args:
            state: Current state as input to the model.
            epsilon: Probability of choosing a random action.

        Returns:
            Tuple of actions (a, b, c).
        """
        if self.action_counter >= self.max_action:
            return None

        self.action_counter += 1
        a = 1
        c = 0.75
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            # Random actions
            # a=2
            # a = 1
            # b = random.uniform(0, 5*np.pi)
            b = random.uniform(0, config.max_x)
            # c = 0.75
            return (a, b, c)

        # Convert state to a tensor, flatten, and add batch dimension
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)

        # Forward pass through the actor-critic model
        a_mean, a_std, b_mean, b_std, c_mean, c_std, _ = self.model(state)

        # Sample actions
        a_dist = torch.distributions.Normal(a_mean, a_std)
        # a = torch.clamp(a_dist.sample(), min = 2, max = 2)
        a = torch.clamp(a_dist.sample(), min=a, max=a)

        c_dist = torch.distributions.Normal(c_mean, c_std)
        c = torch.clamp(c_dist.sample(), min=0.75, max=0.75)

        # Clamp and round `b`
        b_dist = torch.distributions.Normal(b_mean, b_std)
        # b = torch.clamp(b_dist.sample(), min=0, max=5*np.pi)
        b = torch.clamp(b_dist.sample(), min=0, max=config.max_x)

        return (a.item(), b.item(), c.item())

    def get_action_simulate(self, state):
        """
        Get action using the actor-critic model without epsilon-greedy exploration.

        Args:
            state: Current state as input to the model.
            epsilon: Probability of choosing a random action.

        Returns:
            Tuple of actions (a, b, c).
        """
        if self.action_counter >= self.max_action:
            return None

        self.action_counter += 1

        # Convert state to a tensor, flatten, and add batch dimension
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)

        # Forward pass through the actor-critic model
        a_mean, a_std, b_mean, b_std, c_mean, c_std, _ = self.model(state)

        # Sample actions
        a_dist = torch.distributions.Normal(a_mean, a_std)
        a = torch.clamp(a_dist.sample(), min=2, max=2)

        c_dist = torch.distributions.Normal(c_mean, c_std)
        c = torch.clamp(c_dist.sample(), min=0.75, max=0.75)

        # Clamp and round `b`
        b_dist = torch.distributions.Normal(b_mean, b_std)
        b = torch.clamp(b_dist.sample(), min=0, max=5 * np.pi)

        return (a.item(), b.item(), c.item())

    def update(self, state, action, reward, next_state, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)  # Add batch dimension
        next_state = torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Forward pass through the model for current and next states
        a_mean, a_std, b_mean, b_std, c_mean, c_std, value = self.model(state)
        _, _, _, _, _, _, next_value = self.model(next_state)

        # Decompose the action into its components
        a, b, c = action  # These come directly from `get_action` during interaction with the environment

        # Compute TD target for the critic
        target = reward + self.gamma * next_value * (1 - done)
        td_error = target - value  # Temporal Difference error

        # Compute log-probabilities for actions
        # Action `a`: Continuous action modeled by Normal distribution
        a_dist = torch.distributions.Normal(a_mean, a_std)
        log_prob_a = a_dist.log_prob(torch.tensor(a, dtype=torch.float32)).sum()

        # Action `b`: Treated as pseudo-discrete
        b_dist = torch.distributions.Normal(b_mean, b_std)
        log_prob_b = b_dist.log_prob(torch.tensor(b, dtype=torch.float32)).sum()

        # Action `c`: Continuous action modeled by Normal distribution
        c_dist = torch.distributions.Normal(c_mean, c_std)
        log_prob_c = c_dist.log_prob(torch.tensor(c, dtype=torch.float32)).sum()

        # Combine log probabilities
        log_policy = log_prob_a + log_prob_b + log_prob_c

        # Compute losses
        actor_loss = -log_policy * td_error.detach()  # Actor loss: scale log_prob by TD error
        critic_loss = self.criterion(value, target.detach())  # Critic loss: value prediction error

        # Total loss
        total_loss = actor_loss + critic_loss

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def reset_action_counter(self):  # Add a method to reset action counter at the end of each episode
        self.action_counter = 0