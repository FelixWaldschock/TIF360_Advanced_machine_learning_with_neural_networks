import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class DoublePendulumEnv(gym.Env):
    def __init__(self):
        self.gravity = 9.8  # gravitational constant
        self.dt = 0.01  # time step
        self.max_torque = 1.0  # maximum torque
        self.max_speed = 8.0  # maximum angular velocity

        high = np.array([np.pi, self.max_speed, np.pi, self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        th1, th1_dot, th2, th2_dot = self.state

        # equations of motion
        dydt = [
            th1_dot,
            (-3 * self.gravity / (2 * np.pi) * np.sin(th1) - 6 * self.gravity / (2 * np.pi) * np.sin(th1 - 2 * th2)
             - 3 * self.gravity / (2 * np.pi) * np.sin(th1 - 2 * th2) + 6 / (2 * np.pi) ** 2 * th2_dot ** 2 * np.sin(2 * (th1 - th2))
             + 12 / (2 * np.pi) ** 2 * th1_dot ** 2 * np.sin(2 * (th1 - th2))
             + 6 * u / (2 * np.pi) ** 2 * np.cos(th1 - th2)) / (16 - 9 * np.cos(2 * (th1 - th2))),
            th2_dot,
            (9 * self.gravity / (2 * np.pi) * np.sin(th1 - th2) - 6 * self.gravity / (2 * np.pi) * np.sin(th2)
             + 3 * self.gravity / (2 * np.pi) * np.sin(th1 - 2 * th2)
             - 6 / (2 * np.pi) ** 2 * th1_dot ** 2 * np.sin(2 * (th1 - th2))
             + 3 * u / (2 * np.pi) ** 2 * np.cos(th1 - th2)) / (16 - 9 * np.cos(2 * (th1 - th2))),
        ]

        # integrate the equations of motion
        solution = solve_ivp(lambda t, y: dydt, [0, self.dt], [th1, th1_dot, th2, th2_dot])
        self.state = solution.y[:, -1]

        # update state
        th1, th1_dot, th2, th2_dot = self.state
        self.state = np.array([th1, th1_dot, th2, th2_dot])

        # calculate reward
        reward = -(np.cos(th1) + np.cos(th2))

        return self.state, reward, False, {}

    def reset(self):
        high = np.array([np.pi, self.max_speed, np.pi, self.max_speed])
        self.state = np.random.uniform(low=-high, high=high)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        th1, _, th2, _ = self.state
        x1 = np.sin(th1)
        y1 = -np.cos(th1)
        x2 = x1 + np.sin(th2)
        y2 = y1 - np.cos(th2)

        plt.figure()
        plt.plot([0, x1, x2], [0, y1, y2], marker='o', linestyle='-', markersize=10)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Double Pendulum')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    def close(self):
        pass

# Example usage
if __name__ == "__main__":
    env = DoublePendulumEnv()
    observation = env.reset()
    done = False  # Initialize done flag
    while not done:  # Continue until episode is done
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        plt.pause(0.01)  # Pause for a short time to allow the plot to be displayed
        plt.close()  # Close the plot window
    env.close()
