import numpy as np

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_space, mu = 0, sigma=0.4, theta=.01, scale=0.1):
        self.theta = theta
        self.mu = mu*np.ones(action_space)
        self.sigma = sigma
        self.scale = scale
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) + \
            self.sigma * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x * self.scale

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)