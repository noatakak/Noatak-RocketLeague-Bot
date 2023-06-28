import os

import torch

class Agent:
    def __init__(self):
        # If you need to load your model from a file this is the time to do it
        # You can do something like:
        #
        # self.actor = # your Model
        #
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # with open(os.path.join(cur_dir, 'model.p'), 'rb') as file:
        #     model = pickle.load(file)
        # self.actor.load_state_dict(model)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_dir, 'policy_model.pt')

        self.policy: torch.nn.model = torch.jit.load(model_path)
        self.policy.eval()

    def act(self, state):
        # Evaluate your model here
        state = torch.from_numpy(state).float()
        action, _ = self.policy(state)
        action = action.detach().numpy()
        return action
