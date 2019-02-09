import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 16

        self.hidden_1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, input_):
        output_1 = F.relu(self.hidden_1(input_))
        output_2 = F.relu(self.hidden_2(output_1))
        output = self.out(output_2)
        return output


class Actor(nn.Module):

    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        hidden_size = 16

        self.hidden_1 = nn.Linear(self.input_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, self.output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, input_):
        output_1 = F.relu(self.hidden_1(input_))
        output_2 = F.relu(self.hidden_2(output_1))
        output = F.softmax(self.out(output_2), dim=1)
        return output


# Randomly pick an action, with probabilities from the actor network
def get_action(actor, state):
    input_ = torch.FloatTensor(state).reshape([1, 4])

    actor.eval()
    with torch.no_grad():
        output = actor.forward(input_)

    choice = torch.multinomial(output, 1).item()

    # print(f'output: {output} choice: {choice} ')

    return choice


def train_model(critic, actor, prev_states, actions, rewards, curr_states):
    discount_factor = 0.99

    # Get value predictions of the current states, without training
    critic.eval()
    with torch.no_grad():
        predicted_value_current_state = critic.forward(torch.FloatTensor(curr_states)).detach()

    # Since we are finished after the last element, we should predict 0 here
    predicted_value_current_state[-1] = 0

    # Estimate the 'real' value as the reward + the (discounted) predicted value of the current state
    real_previous_value = torch.FloatTensor(rewards).reshape(-1, 1) + discount_factor * predicted_value_current_state

    prev_states_tensor = torch.FloatTensor(prev_states)

    # Forward propagate to get value predictions for the previous state
    critic.train()
    predicted_value_previous_state = critic.forward(prev_states_tensor)

    # Forward propagate to get action probabilities for the previous state
    actor.train()
    actor_output = actor.forward(prev_states_tensor)

    # Calculate the "advantage" -- how much better/worse we did than what we predicted
    advantage = (real_previous_value - predicted_value_previous_state.detach())

    # Decide what would have been the best action
    def best_action(action, advantage_):
        if action is 0:
            return 0 if advantage_ > 0 else 1
        else:
            return 1 if advantage_ > 0 else 0

    # Get the action to learn for every pair of action and advantage
    action_to_learn = torch.LongTensor(list(
        map(lambda x: best_action(x[0], x[1]), zip(actions, advantage))))

    # train the actor
    actor_loss = F.cross_entropy(actor_output, action_to_learn)
    actor.optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm(actor.parameters(), 0.5)
    actor.optimizer.step()

    # train the critic
    critic_loss = F.mse_loss(predicted_value_previous_state, real_previous_value)
    critic.optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm(critic.parameters(), 0.5)
    critic.optimizer.step()

    # for i in range(len(actions)):
    #     print(f'action: {actions[i]} reward: {rewards[i]} advantage: {advantage[i].item():.2f}'
    #           f' actor_output: {actor_output[i].data[:]} action_to_learn: {action_to_learn[i]}'
    #           f' critic_output: {predicted_value_previous_state[i].item():.2f}'
    #           f' critic_to_learn: {real_previous_value[i].item():.2f}')


# After drawing plot, continue running while not stealing focus
def release_plot(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        fig_manager = matplotlib._pylab_helpers.Gcf.get_active()
        if fig_manager is not None:
            canvas = fig_manager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)


def train():
    render = False
    draw_plot = False

    # Number of max scores in a row that we regard as success
    success_count = 10

    env = gym.make('CartPole-v1')

    # In CartPole the maximum length of an episode is 500
    max_score = 500

    # Get sizes of state and action from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor = Actor(input_size=state_size, output_size=action_size)
    critic = Critic(input_size=state_size, output_size=1)

    scores, episodes = [], []

    episode = 0
    all_done = False

    if draw_plot:
        plt.ion()
        plt.show()

    while not all_done:
        env_done = False
        score = 0
        state = env.reset()

        states, next_states, actions, rewards = [], [], [], []

        # Run one episode, i.e. until we lose or succeed
        while not env_done:
            if render:
                env.render()

            action = get_action(actor, state)

            # Perform an action in the environment
            next_state, reward, env_done, info = env.step(action)
            score += reward

            # If an action makes the episode end, give it a penalty
            reward_mod = reward if (not env_done or score == max_score) else -5

            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward_mod)

            state = next_state

            if env_done:
                train_model(critic, actor, states, actions, rewards, next_states)

                scores.append(score)
                episodes.append(episode)

                if draw_plot:
                    plt.plot(episodes, scores, 'b')
                    release_plot(1)

                print('episode:', episode, ' score:', score)

                # If we have success_count perfect runs in a row, stop training
                if len(scores) >= success_count and np.mean(scores[-success_count:]) == max_score:
                    all_done = True

        episode += 1

    env.close()


if __name__ == '__main__':
    train()
