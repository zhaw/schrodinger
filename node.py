import mxnet as mx
import numpy as np

class Node(object):
    def __init__(self):
        pass

    def register_parent(self, upper, ith_child):
        self.parent = upper
        self.prob_in = upper.outputs[ith_child]
        upper.children[ith_child] = self

    def alloc_children(self):
        self.outputs = [] 
        self.grad = []
        self.children = []
        for i in range(self.n_children):
            tmp = []
            gtmp = []
            for j in range(n_players):
                tmp.append(np.ones([1, n_states]) * 1./n_states)
                gtmp.append(np.ones([1, n_states]) * 1./n_states)
            self.outputs.append(tmp)
            self.grad.append(gtmp)
            self.children.append([])

    def forward(self):
        pass

    def backward(self):
        pass


class RootNode(Node):
    def __init__(self, n_players, n_states, name):
        self.probs = []
        self.n_children = 1
        for j in range(n_players):
            self.probs.appned(np.ones([1, n_states]) * 1./n_states)
        super(RootNode, self).alloc_children()

    def forward(self):
        pass

    def backward(self):
        pass



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
        super(DecisionNode, self).register_parent(upper, ith_child)
        self.n_children = n_children
        super(DecisionNode, self).alloc_children(self)
        self.get_policy_maker()
        self.make_graph()
        self.player = ith_player

    def get_policy_maker(self):
        pass

    def make_graph(self):

    def forward(self):
        '''
        FORWARD THE POLICY MAKER AND GET THE PROB OF EACH MOVE
        COMPUTE PROBABILITY AT EACH CHILDREN NODE
        ACCUMULATE COST
        '''
        policy = self.policy_maker.forward()

    def backward(self):
        '''
        BACKPROPAGATE THE DECISION MAKER
        PASS THE GRADIENT UP
        '''
        pass


class RevealNode(Node):
# WHAT SHOULD WE DO AT RevealNode?
    # RevealNode has only one child
    # When forward, RevealNode provide a sample of board states
    # When backward, RevealNode does nothing
    def __init__(self, upper, ith_child, name):
        super(RevealNode, self).address_children(upper, ith_child)

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
        super(AwardNode, self).address_children(upper, ith_child)
        self.winner = winner 

class ShowdownNode(Node):
    def __init__(self, upper, ith_child, name):
        super(ShowdownNode, self).address_children(upper, ith_child)

    def forward(self):
        '''
        COMPUTE AWARD ACCORDING TO DISTRIBUTION OF TWO PLAYERS HAND
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

