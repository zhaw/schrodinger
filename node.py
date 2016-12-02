import mxnet as mx
import numpy as np

N_STATES = 13
N_PLAYERS = 2

class PolicyMaker():
    # PolicyMaker needs to return n_choices nd-array of shape (N_STATES,).
    def __init__(self, n_choices, feature, name):
        self.name = name
        self.n_choices = n_choices
        self.args = [mx.sym.Variable('%s_pm'%name)]
        self.out = mx.sym.SoftmaxActivation(self.args[0], mode='instance') # n_states * n_choices
        self.out = mx.sym.SliceChannel(self.out, num_outputs=n_choices, axis=1, squeeze_axis=True)

    def infer_shape(self):
        self.shapes = [(N_STATES, self.n_choices)]


class Node(object):
    def __init__(self):
        pass

    def register_parent(self, upper, ith_child):
        self.parent = upper
        self.pin = upper.outputs[ith_child]
        upper.children[ith_child] = self
        self.pot = upper.pot + upper.cost[ith_child]
        self.board = upper.board + []

    def alloc_children(self):
        self.pout = [] 
        self.children = []
        for i in range(self.n_children):
            tmp = []
            for j in range(N_PLAYERS):
                tmp.append([])
            self.pout.append(tmp)
            self.children.append([])


class RootNode(Node):
    def __init__(self, n_states, name):
        self.pin = []
        self.n_children = 1
        for j in range(N_PLAYERS):
            self.pin.append(mx.sym.Variable('player%d_prob'%j)) # should be np.ones([1, n_states]) * 1./n_states
        super(RootNode, self).alloc_children()
        for j in range(N_PLAYERS):
            self.pout[j] = self.pin
        self.board = []
        self.pot = 0
        self.loss = [0 for i in range(N_PLAYERS)]


class DecisionNode(Node):
    def __init__(self, upper, ith_child, ith_player, n_children, cost, name):
        '''
        upper:          parent node
        ith_child:      parent's which child
        ith_player:     who's turn to make decision
        n_children:     how many children does it have
        cost:           cost of each decision
        name:           node's name
        '''
        self.name = name
        self.cost = cost
        self.player = ith_player
        super(DecisionNode, self).register_parent(upper, ith_child)
        self.n_children = n_children
        self.policy_maker = PolicyMaker(n_children, [], self.name) 
        super(DecisionNode, self).alloc_children(self)
        for i in range(self.n_children):
            for j in range(N_PLAYERS):
                if j == self.player:
                    self.pout[j] = self.policy_maker.out[i] * self.pin[j]
                else:
                    self.pout[j] = self.pin[j]

    def update_feature(self):
        pass


class RevealNode(Node):
# WHAT SHOULD WE DO AT RevealNode?
    # RevealNode has only one child
    # When forward, RevealNode provide a sample of board states
    # When backward, RevealNode does nothing
    def __init__(self, upper, ith_child, name):
        self.cost = [0] * N_PLAYERS
        super(RevealNode, self).register_parent(upper, ith_child)
        super(RevealNode, self).address_children(upper, ith_child)
        for j in range(N_PLAYERS):
            self.pout[j] = self.pin

    def forward(self):
        '''
        SET BOARD FEATURE AND PASS THE PROBS AND POT TO CHILDREN NODES
        '''
        pass

    def backward(self):
        '''
        JUST PASS GRADIENT FROM DOWN NODE UP
        '''
        pass


class AwardNode(Node):
    def __init__(self, upper, ith_child, winner, name):
        self.cost = [0] * N_PLAYERS
        super(AwardNode, self).address_children(upper, ith_child)
        self.winner = winner 

class ShowdownNode(Node):
    def __init__(self, upper, ith_child, name):
        self.cost = [0] * N_PLAYERS
        super(ShowdownNode, self).address_children(upper, ith_child)

    def forward(self):
        '''
        COMPUTE AWARD ACCORDING TO DISTRIBUTION OF TWO PLAYERS HAND AND THE BOARD
        '''
        pass

    def backward(self):
        '''
        COMPUTE AWARD'(p) FOR POSSIBILITY OF EACH POSSIBLE HAND
        '''


root = RootNode(2, 52*51/2, 'root') # init stats, probs of players hands
pf1d = DecisionNode(root, 0, 'pf1d')
pf1c = RevealNode(root, 1, 'pf1c')
pf1f = RewardNode(root, 2, 'pf1f')

