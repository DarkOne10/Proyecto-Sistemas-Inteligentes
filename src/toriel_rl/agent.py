import json
from pathlib import Path

import numpy as np


class QLearningAgent:
    """Tabular Q-learning over position, HP, attack, Spare count, and nearby danger."""

    position_edges = (0.33, 0.66)
    vertical_edges = (0.33, 0.66)
    hp_edges = (0.33, 0.66)
    phase_edges = (0.5,)

    def __init__(
        self,
        action_count: int = 5,
        learning_rate: float = 0.15,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ) -> None:
        if action_count <= 0:
            raise ValueError("Action count must be positive.")
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1].")
        if not 0 <= discount_factor <= 1:
            raise ValueError("Discount factor must be in [0, 1].")
        if not 0 <= epsilon_min <= epsilon <= 1:
            raise ValueError("Epsilon values must satisfy 0 <= min <= epsilon <= 1.")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("Epsilon decay must be in (0, 1].")

        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[tuple[int, int, int], np.ndarray] = {}
        self.rng = np.random.default_rng(seed)

    @classmethod
    def discretize(cls, observation: np.ndarray) -> tuple[int, int, int, int, int, int]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (12,):
            raise ValueError("Expected a twelve-value TorielEnv observation.")

        position_x = float(values[6]) if values[3] == 0.0 else float(values[0])
        position_x_bin = int(np.digitize(position_x, cls.position_edges))
        position_y_bin = int(np.digitize(float(values[1]), cls.vertical_edges))
        hp_bin = int(np.digitize(float(values[2]), cls.hp_edges))
        attack_bin = int(round(float(values[3]) * 3))
        phase_bin = int(np.digitize(float(values[4]), cls.phase_edges))
        danger_mask = sum(1 << index for index, value in enumerate(values[8:12]) if value <= 0.08)
        return position_x_bin, position_y_bin, hp_bin, attack_bin, phase_bin, danger_mask

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))
        return int(np.argmax(self._values_for(self.discretize(observation))))

    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not 0 <= action < self.action_count:
            raise ValueError(f"Invalid action: {action}")

        state = self.discretize(observation)
        next_state = self.discretize(next_observation)
        old_value = self._values_for(state)[action]
        done = terminated or truncated
        next_value = 0.0 if done else float(np.max(self._values_for(next_state)))
        target = reward + self.discount_factor * next_value
        self._values_for(state)[action] = old_value + self.learning_rate * (target - old_value)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_version": 4,
            "action_count": self.action_count,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_table": [
                {"state": list(state), "values": values.tolist()}
                for state, values in sorted(self.q_table.items())
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("state_version") != 4:
            raise ValueError("This Q-table uses an old observation format; retrain the agent.")
        agent = cls(
            action_count=int(payload["action_count"]),
            learning_rate=float(payload["learning_rate"]),
            discount_factor=float(payload["discount_factor"]),
            epsilon=float(payload["epsilon"]),
            epsilon_min=float(payload["epsilon_min"]),
            epsilon_decay=float(payload["epsilon_decay"]),
        )
        agent.q_table = {
            tuple(int(value) for value in row["state"]): np.asarray(
                row["values"], dtype=np.float64
            )
            for row in payload["q_table"]
        }
        if any(values.shape != (agent.action_count,) for values in agent.q_table.values()):
            raise ValueError("Q-table contains an invalid action vector.")
        return agent

    def _values_for(self, state: tuple[int, int, int]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_count, dtype=np.float64)
        return self.q_table[state]