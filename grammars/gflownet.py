from abc import ABC, abstractmethod


class GFN_controller(ABC):
    def __init__(self, device, args):
        self.device = device
        self.args = args

    @abstractmethod
    def batch_calc_forward_prob(self, F_logits, F_action):
        pass

    @abstractmethod
    def apply_forward_actions(self, state, F_action):
        pass

    @abstractmethod
    def reverse_forward_actions(self, state, F_action):
        pass

    @abstractmethod
    def sample_forward(self, F_logits, state, greedy, temperature):
        pass

    @abstractmethod
    def batch_calc_backward_prob(self, B_logits, B_action):
        pass

    @abstractmethod
    def apply_backward_actions(self, state, B_action):
        pass

    @abstractmethod
    def reverse_backward_actions(self, state, B_action):
        pass

    @abstractmethod
    def sample_backward(self, B_logits, state, greedy, temperature):
        pass


from copy import deepcopy


class GFN_state(ABC):
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self._state = None

    def clone(self):
        # VERY SLOW!!
        return deepcopy(self)

    @abstractmethod
    def from_iterable(self, iter):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __eq__(self):
        pass
