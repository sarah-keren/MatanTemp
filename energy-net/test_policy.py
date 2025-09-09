import numpy as np
from pathlib import Path

from stable_baselines3 import TD3

from energy_net.controllers.alternating_wrappers import make_pcs_env


MODEL_PATH = Path("models/ext/td3_pcs_best.zip")
NORM_PATH = Path("models/ext/td3_pcs_best_norm.pkl")


def collect_actions(n_steps: int = 5000):
    env = make_pcs_env(
        steps_per_iteration=4800,
        cost_type="CONSTANT",
        pricing_policy="ONLINE",
        demand_pattern="DATA_DRIVEN",
        demand_data_path="configs/demand_data_sample.yaml",
        use_dispatch_action=True,
        eval_mode=True,
        norm_path=str(NORM_PATH),
    )
    model = TD3.load(str(MODEL_PATH))
    model.set_env(env)
    obs = env.reset()
    actions = []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        actions.append(action[0])
        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()
    return np.array(actions)


def test_td3_pcs_best_not_constant_one():
    actions = collect_actions()
    # Fail the test if every action equals 1.0
    assert not np.allclose(actions[500:], 1.0), "PCS model output is constant 1"



if __name__ == "__main__":
    test_td3_pcs_best_not_constant_one()