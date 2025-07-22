import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from PIL import Image

class DrawingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, image_path, render_mode=None):
        super(DrawingEnv, self).__init__()

        self.ref_image = self._load_and_process_image(image_path)
        self.canvas_shape = self.ref_image.shape[:2]
        self.canvas = np.zeros(self.canvas_shape, dtype=np.uint8)

        # Action space: [pen_action, dx, dy]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation space: [x, y, last_dx, last_dy] + flattened downsampled canvas
        self.canvas_downsampled_shape = (32, 32)
        self.observation_space = spaces.Dict({
            'pen_state': spaces.Box(
                low=np.array([0, 0, -1, -1]),
                high=np.array([self.canvas_shape[1], self.canvas_shape[0], 1, 1]),
                dtype=np.float32
            ),
            'canvas': spaces.Box(
                low=0, high=255,
                shape=self.canvas_downsampled_shape,
                dtype=np.uint8
            )
        })

        self.pen_pos = np.array([self.canvas_shape[1] / 2, self.canvas_shape[0] / 2], dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.pen_down = True

        self.reward_weights = {
            'likeness': 1.0,
            'boundary': -10.0,
            'smoothness': -0.5
        }
        self.path_history = []
        self.path_history.append(self.pen_pos.copy())


    def _load_and_process_image(self, image_path):
        try:
            pil_image = Image.open(image_path).convert('L')
            image = np.array(pil_image)
        except Exception as e:
            raise IOError(f"Could not load image from {image_path} using Pillow. Error: {e}")

        if image is None:
            raise ValueError(f"Could not convert image from {image_path} to numpy array.")

        # For simplicity, we'll use a Canny edge map as the reference
        edges = cv2.Canny(image, 100, 200)
        resized_edges = cv2.resize(edges, (256, 256))
        return resized_edges

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.canvas.fill(0)
        self.pen_pos = np.array([self.canvas_shape[1] / 2, self.canvas_shape[0] / 2], dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.pen_down = True
        self.path_history = [self.pen_pos.copy()]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Discretize the first action value to be pen up/down
        pen_action = action[0]
        move_action = action[1:]
        
        self.pen_down = pen_action > 0

        prev_pos = self.pen_pos.copy()

        # Update pen position based on action (velocity)
        self.pen_pos += move_action * 5 # Scale action to have a larger effect

        # Clip to stay within canvas boundaries
        self.pen_pos[0] = np.clip(self.pen_pos[0], 0, self.canvas_shape[1] - 1)
        self.pen_pos[1] = np.clip(self.pen_pos[1], 0, self.canvas_shape[0] - 1)
        
        self.path_history.append(self.pen_pos.copy())

        # Draw line on canvas if pen is down
        if self.pen_down:
            cv2.line(self.canvas,
                     (int(prev_pos[0]), int(prev_pos[1])),
                     (int(self.pen_pos[0]), int(self.pen_pos[1])),
                     255, 1)

        # Calculate reward
        reward = self._calculate_reward(prev_pos, self.pen_pos)

        # Check for termination
        terminated = self._check_if_done()
        truncated = False # For now, we don't have a truncation condition

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        downsampled_canvas = cv2.resize(self.canvas, self.canvas_downsampled_shape)
        return {
            'pen_state': np.concatenate([self.pen_pos, self.last_action]),
            'canvas': downsampled_canvas
        }

    def _get_info(self):
        return {}

    def _calculate_reward(self, prev_pos, current_pos):
        likeness_reward = 0

        if self.pen_down:
            # Reward for drawing on edges
            line_mask = np.zeros_like(self.canvas)
            cv2.line(line_mask,
                     (int(prev_pos[0]), int(prev_pos[1])),
                     (int(current_pos[0]), int(current_pos[1])),
                     255, 1)

            intersection = np.logical_and(line_mask, self.ref_image)
            likeness_reward = np.sum(intersection) / 255.0 # Normalize by pixel value

        # Penalty for hitting the boundary
        boundary_penalty = 0
        if (current_pos[0] == 0 or current_pos[0] == self.canvas_shape[1] - 1 or
            current_pos[1] == 0 or current_pos[1] == self.canvas_shape[0] - 1):
            boundary_penalty = self.reward_weights['boundary']

        # Penalty for high curvature
        smoothness_penalty = 0
        if self.pen_down and len(self.path_history) >= 3:
            p1 = self.path_history[-3]
            p2 = self.path_history[-2]
            p3 = self.path_history[-1]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Normalize vectors
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
            
            # Calculate dot product
            dot_product = np.dot(v1_norm, v2_norm)
            
            # The penalty is based on the change in angle.
            # A dot product of 1 means no change, -1 means 180 degree turn.
            # We penalize sharp turns, so we use (1 - dot_product).
            angle_penalty = (1 - dot_product)
            smoothness_penalty = self.reward_weights['smoothness'] * angle_penalty


        return self.reward_weights['likeness'] * likeness_reward + boundary_penalty + smoothness_penalty

    def _check_if_done(self):
        # For now, we'll just run for a fixed number of steps, handled by the training loop
        return False

    def render(self):
        # For now, we'll just return the canvas
        return self.canvas

    def close(self):
        pass