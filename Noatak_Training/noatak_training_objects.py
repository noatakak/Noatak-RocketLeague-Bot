import numpy as np

from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BALL_RADIUS, BACK_NET_Y
from rlgym.utils.gamestates import GameState, PlayerData

from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards import SaveBoostReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import \
    LiuDistanceBallToGoalReward, VelocityBallToGoalReward, BallYCoordinateReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import \
    LiuDistancePlayerToBallReward, TouchBallReward, FaceBallReward, VelocityPlayerToBallReward

from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards
from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards
from rlgym_tools.extra_rewards.multiply_rewards import MultiplyRewards




class NoatakReward(RewardFunction):
    """
    A combination of lots of different reward functions, using AnnealRewards to start with the simpler goals
    before moving to the more complicated ones, MultiplyRewards to combine multiple reward functions at the same time,
    and DistributeRewards to make sure that the reward is Distributed to the correct agents.
    """
    def __init__(self, rewardTransition: int):
        super().__init__()
        # Define general EventReward
        self.generalReward = EventReward(goal=1, team_goal=0.5, concede=-1, touch=.5, shot=0.5, save=1,
                                         demo=0.25, boost_pickup=0.5)
        self.saveBoost = SaveBoostReward()

        # Section 1: Train to approach ball
        self.section1Functions = [self.generalReward, self.saveBoost, FaceBallReward(), TouchBallReward(), VelocityPlayerToBallReward(), LiuDistancePlayerToBallReward()]
        self.section1Weights = [0.25, .25, 0.75, 2, 1, 2]
        self.section1Reward = CombinedReward(reward_functions=self.section1Functions,
                                             reward_weights=self.section1Weights)

        # Section 2: Train to get ball in the air
        # self.section2Functions = [self.generalReward, BallYCoordinateReward(), TouchBallReward(aerial_weight=.5)]
        # self.section2Weights = [0.5, 0.75, 0.5]
        # self.section2Reward = CombinedReward(reward_functions=self.section2Functions,
        #                                      reward_weights=self.section2Weights)

        # Section 3: Train to get ball in opponent goal
        self.section3Functions = [self.generalReward, self.saveBoost, LiuDistanceBallToGoalReward(), VelocityBallToGoalReward()]
        self.section3Weights = [1, .25, 2, 1]
        self.section3Reward = CombinedReward(reward_functions=self.section3Functions,
                                             reward_weights=self.section3Weights)

        # List of reward steps
        alternating_reward_steps = [self.section1Reward, rewardTransition, self.section3Reward]

        # Transition between reward functions, set mode to STEP, TOUCH, or GOAL
        self.anneal = AnnealRewards(*alternating_reward_steps, mode=AnnealRewards.STEP)

        # Distribute rewards to all agents
        # self.distributor = DistributeRewards(reward_func=self.anneal, team_spirit=0.3)

    def reset(self, initial_state: GameState):
        # self.distributor.reset(initial_state=initial_state)
        self.anneal.reset(initial_state=initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        #return self.distributor.get_reward(player=player, state=state, previous_action=previous_action)
        if self.anneal.next_transition_step == self.anneal.last_transition_step:
            self.anneal.next_transition_step += 1
        return self.anneal.get_reward(player=player, state=state, previous_action=previous_action)
