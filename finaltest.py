import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

# Use the same Env class as during training
class Env:
    def __init__(self, img_stack=4, action_repeat=8):
        self.env = gym.make('CarRacing-v3', render_mode="rgb_array")
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = img_stack
        self.action_repeat = action_repeat

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        self.die = False

        obs, _ = self.env.reset()  # Handle tuple return
        img_rgb = obs
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for _ in range(self.action_repeat):
            img_rgb, reward, terminated, truncated, _ = self.env.step(action)
            die = terminated or truncated

            if die:
                reward += 100
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05

            total_reward += reward
            done = True if self.av_r(reward) <= -0.1 else False

            if done or die:
                break

        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        return self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        if norm:
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

# Actor-Critic Network for PPO
class Net(nn.Module):
    def __init__(self, img_stack):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v

# Agent class for testing
class Agent:
    def __init__(self, img_stack):
        self.net = Net(img_stack).float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)
        return action.squeeze().cpu().numpy()

    def load_param(self, path='param/ppo_net_params1.pkl'):
        self.net.load_state_dict(torch.load(path, map_location=device))

# Initialize and test
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_stack = 4
    action_repeat = 8

    agent = Agent(img_stack)
    agent.load_param()
    env = Env(img_stack=img_stack, action_repeat=action_repeat)

    state = env.reset()
    for i_ep in range(1):  # Adjust for multiple episodes
        score = 0
        state = env.reset()
        frames = []  # For saving video frames
        for t in range(1000):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            frame = env.render()
            frames.append(frame)
            score += reward
            state = state_
            if done or die:
                break

        print(f'Episode {i_ep}\tScore: {score:.2f}')

    # Save video
    import imageio
    imageio.mimsave("car_racing_test.mp4", frames, fps=30)
    print("Video saved as car_racing_test.mp4")
