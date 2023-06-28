import os
import torch
from datetime import datetime

from Noatak_Training.policy import Policy

from stable_baselines3 import PPO

from rlgym_tools.sb3_utils import SB3SingleInstanceEnv
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from rlgym.envs import Match
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import LiuDistanceBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState, RandomState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition

from Noatak_Training.noatak_training_objects import NoatakReward

# Set training time and section lengths
num_steps = 1000000000
section_steps = (num_steps/1)
reset_seconds = 30
rl_instances = 10

# choose to load previous trial
input_path = ""

# choose where to save current trial
output_path = "1BilSteps"


def get_match():
    # Calcualte steps for max seconds
    default_tick_skip = 8
    physics_ticks_per_second = 120
    max_steps = int(round(reset_seconds * physics_ticks_per_second / default_tick_skip))

    # Define terminal conditions
    condition1 = GoalScoredCondition()
    condition2 = TimeoutCondition(max_steps)

    # Define custom match set-up
    return Match(
        reward_function=NoatakReward(rewardTransition=section_steps),
        terminal_conditions=[condition1, condition2],
        obs_builder=DefaultObs(),
        state_setter=RandomState(),
        action_parser=DefaultAction(),
        spawn_opponents=True,
        team_size=1,
        game_speed=1000
    )

# Run this in mutliple terminals to train different models at the same time
# "& c:/Users/noata/Noatak-RocketLeague-Bot/venv/Scripts/python.exe c:/Users/noata/Noatak-RocketLeague-Bot/main.py"
def main():
    # Import policy if input is defined.
    if input_path != "":
        print("Loading model from file...")
        try:
            policy: Policy = torch.jit.load("C:/Users/noata/Noatak-RocketLeague-Bot/Noatak_Training/PolicyModels/" + input_path + "/policy_model.pt")
        except Exception as e:
            print("Error loading model from file. Aborting.")
            raise e
        
    
    # Intiialize the environment
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=rl_instances, wait_time=20)
    
    # Define the model
    model = PPO(policy="MlpPolicy", env=env, verbose=1, device='cpu')

    # Inject policy into model (if using input file)
    if input_path != "":
        print("Injecting model into PPO...")
        try:
            model.policy.mlp_extractor = policy.extractor
            model.policy.action_net = policy.action_net
            model.policy.value_net = policy.value_net
        except Exception as e:
            print("Error injecting model. Cancelling training.")
            raise e
    
    # Train the model
    model.learn(total_timesteps=int(num_steps), progress_bar=True)

    # Save the model 
    policy = Policy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net).to('cpu')
    model_scripted = torch.jit.script(policy)

    output_path = output_path + datetime.now().strftime("_D%m%d-T%H%M")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if input_path == "":
        os.mkdir(os.path.join(cur_dir, 'Noatak_Training', 'PolicyModels', output_path))
    model_path = os.path.join(cur_dir, 'Noatak_Training', 'PolicyModels', output_path, 'policy_model.pt')
    print("saving model to location: " + model_path)

    model_scripted.save(model_path)

    env.close()


if __name__ == "__main__":
    main()
