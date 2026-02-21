import torch

FOLD    = "FOLD"
CALL    = "CALL"
RAISE_2 = "RAISE_2"
RAISE_4 = "RAISE_4"
ALLIN   = "ALLIN"

ALL_ACTIONS      = [FOLD, CALL, RAISE_2, RAISE_4, ALLIN]

class CFRDatasetCollector:
    """
    Collects:
        state_features
        nash_policy
        state_value
    during CFR traversal.
    """

    def __init__(self, encode_state_fn):
        self.encode_state = encode_state_fn
        self.samples = []

    def __call__(self, state, sigma, value):
        features = self.encode_state(state)

        policy = torch.tensor(
            [sigma.get(a, 0.0) for a in ALL_ACTIONS],
            dtype=torch.float32,
        )

        value = torch.tensor(value, dtype=torch.float32)
        self.samples.append((features, policy, value))

    def get_dataset(self):
        return self.samples
