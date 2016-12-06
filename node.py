import mxnet as mx
import numpy as np

CTX = mx.gpu()

N_STATES = 13
N_PLAYERS = 2
N_BOARD = 1

CARDS = [(i,j) for i in range(2) for j in range(13)]

BATCH_SIZE = 320 
LR = 1e-1
WD = 1e-4
MOMENTUM = 0.9

class PolicyMaker():
    # PolicyMaker needs to return n_children nd scalar
    def __init__(self, n_children, feature, node):
        self.name = node.name
        self.node = node
        self.n_children = n_children
        data = mx.sym.Concat(*feature, num_args=len(feature))
        flat = mx.sym.Flatten(data)
        hid1 = mx.sym.FullyConnected(data=flat, num_hidden=5, name='%s_fc1'%node.name, no_bias=True)
        act1 = mx.sym.Activation(data=hid1, act_type='sigmoid')
        hid2 = mx.sym.FullyConnected(data=act1, num_hidden=self.n_children, name='%s_fc2'%node.name, no_bias=True)
        self.out = mx.sym.SoftmaxActivation(hid2, mode='instance') # of shape (BATCH_SIZE, n_children)
        self.out = mx.sym.SliceChannel(self.out, num_outputs=n_children, axis=1, squeeze_axis=True, name='slicechannel_%s'%node.name)
        self.arg_names = []
        self.arg_nds = []
        for i in range(2):
            self.arg_names.append('%s_fc%d_weight'%(node.name, i+1))
#            self.arg_names.append('%s_fc%d_bias'%(node.name, i+1))
        self.arg_nds.append(mx.nd.zeros([5, len(feature)*2*13], CTX))
#        self.arg_nds.append(mx.nd.zeros([80], CTX))
        self.arg_nds.append(mx.nd.zeros([self.n_children, 5], CTX))
#        self.arg_nds.append(mx.nd.zeros([self.n_children], CTX))
        self.arg_dict = dict([(name, nd) for name, nd in zip(self.arg_names, self.arg_nds)])

        initializer = mx.init.Normal(1e-3)
        self.grad_dict = {}
        for k in self.arg_dict:
            initializer(k, self.arg_dict[k])
            self.grad_dict[k] = mx.nd.zeros(self.arg_dict[k].shape, CTX)

    def get_executor(self, arg_dict):
        for k in self.out.list_arguments():
            if not k in self.arg_names:
                self.arg_dict[k] = arg_dict[k]
        self.executor = self.out.bind(ctx=CTX, args=self.arg_dict, args_grad=self.grad_dict, grad_req='write')
        self.optimizer = mx.optimizer.SGD(learning_rate=LR, wd=WD, momentum=MOMENTUM)
        self.optim_states = []
        for i, var in enumerate(self.executor.grad_dict):
            if var[1] == '+':
                self.optim_states.append(self.optimizer.create_state(i, self.executor.arg_dict[var]))
            else:
                self.optim_states.append([])

    def forward(self, graph):
        self.executor.forward(is_train=True)
        for i in range(self.n_children):
            self.executor.outputs[i].copyto(graph.arg_dict['%s_%d_policy'%(self.name,i)])

    def update(self, grad_list):
        s = mx.nd.zeros([1], CTX)
        for v in grad_list:
            s += mx.nd.sum(mx.nd.abs(v))
        s = mx.nd.maximum(s, mx.nd.ones([1], CTX)*1e-30)
        s = s.asnumpy()[0]
        self.executor.backward([-v/s for v in grad_list])
        for i, k in enumerate(self.executor.grad_dict):
            if k[1] == '+':
                self.optimizer.update(i, self.executor.arg_dict[k], self.executor.grad_dict[k], self.optim_states[i])

    def dump(self, folder):
        mx.nd.save('%s/%s.nd'%(folder, self.name), self.executor.arg_dict)

    def load(self, folder):
        pretrained = mx.nd.load('%s/%s.nd'%(folder, self.name))
        for k in pretrained:
            pretrained[k].copyto(self.executor.arg_dict[k])


class Node(object):
    def __init__(self):
        pass

    def register_parent(self, upper, ith_child):
        self.parent = upper
        self.pin = upper.pout[ith_child]
        upper.children[ith_child] = self
        self.pot = upper.pot + upper.cost[ith_child]
        self.loss = upper.loss + []
        try:
            # parent is decision_node only if it has player attribute
            self.loss[upper.player] += upper.cost[ith_child]
        except:
            pass
        self.feature = upper.feature
        self.visible = upper.visible + []

    def alloc_children(self):
        self.pout = [] 
        self.children = []
        for i in range(self.n_children):
            self.pout.append([])
            self.children.append([])


class RootNode(Node):
    def __init__(self, name):
        self.name = name
        self.pin = mx.sym.Variable('root_p') # mx.nd.array of shape (1) and is constantly 1
        self.n_children = 1
        super(RootNode, self).alloc_children()
        self.pout[0] = 1 * self.pin
        self.feature = []
        self.arg_names = []
        self.arg_nds = []
        self.arg_names.append('root_p')
        self.arg_nds.append(mx.nd.ones([BATCH_SIZE], CTX))
        for j in range(N_PLAYERS):
            self.feature.append(mx.sym.Variable('player%d_hand'%j))
            self.arg_names.append('player%d_hand'%j)
            self.arg_nds.append(mx.nd.zeros([BATCH_SIZE,1,2,13], CTX))
        for j in range(N_BOARD):
            self.feature.append(mx.sym.Variable('board%d'%j))
            self.arg_names.append('board%d'%j)
            self.arg_nds.append(mx.nd.zeros([BATCH_SIZE,1,2,13], CTX))
        self.pot = 1
        self.loss = [0 for i in range(N_PLAYERS)]
        self.loss[1] = 1
        self.cost = [0]
        self.visible = []
        for j in range(N_PLAYERS):
            self.visible.append(1)
        for j in range(N_BOARD):
            self.visible.append(0)
        self.arg_dict = dict([(name, nd) for name, nd in zip(self.arg_names, self.arg_nds)])


    def sample(self):
        def get_win_share(s):
            if s[0][1] == s[2][1]:
                return [1, 0]
            if s[1][1] == s[2][1]:
                return [0, 1]
            if s[0][1] == s[1][1]:
                return [0.5, 0.5]
            if s[0][1] > s[1][1]:
                return [1, 0]
            if s[0][1] < s[1][1]:
                return [0, 1]

        samples = []
        cc0 = []
        cc1 = []
        cc2 = []
        for i in range(BATCH_SIZE):
            x = np.random.choice(range(len(CARDS)), N_PLAYERS+N_BOARD, replace=False)
#            x[0] = i%26
#            x[1] = 3
#            x[2] = 2
            cc0.append(CARDS[x[0]][1])
            cc1.append(CARDS[x[1]][1])
            cc2.append(CARDS[x[2]][1])
            tmp = []
            for xx in x:
                tmp.append(CARDS[xx])
            samples.append(tmp)
        for j in range(N_PLAYERS):
            tmp = np.zeros(self.arg_dict['player%d_hand'%j].shape)
            for i in range(BATCH_SIZE):
                for k in range(1):
                    tmp[i, k, samples[i][j][0], samples[i][j][1]] = 1
            self.arg_dict['player%d_hand'%j][:] = tmp
        for j in range(N_BOARD):
            tmp = np.zeros(self.arg_dict['board%d'%j].shape)
            for i in range(BATCH_SIZE):
                for k in range(1):
                    tmp[i, k, samples[i][j+N_PLAYERS][0], samples[i][j+N_PLAYERS][1]] = 1
            self.arg_dict['board%d'%j][:] = tmp
        return np.array(map(get_win_share, samples)), cc0, cc1, cc2
    

    def manual(self):
        def get_win_share(s):
            if s[0][1] == s[2][1]:
                return [1, 0]
            if s[1][1] == s[2][1]:
                return [0, 1]
            if s[0][1] == s[1][1]:
                return [0.5, 0.5]
            if s[0][1] > s[1][1]:
                return [1, 0]
            if s[0][1] < s[1][1]:
                return [0, 1]
        code = eval(raw_input("cards?\n"))
        samples = [code for i in range(BATCH_SIZE)]
        for j in range(N_PLAYERS):
            tmp = np.zeros(self.arg_dict['player%d_hand'%j].shape)
            for k in range(1):
                tmp[0, k, code[j][0], code[j][1]] = 1
            self.arg_dict['player%d_hand'%j][:] = tmp
        for j in range(N_BOARD):
            tmp = np.zeros(self.arg_dict['board%d'%j].shape)
            for k in range(1):
                tmp[0, k, code[j+N_PLAYERS][0], code[j+N_PLAYERS][1]] = 1
            self.arg_dict['board%d'%j][:] = tmp
        return np.array(map(get_win_share, samples))


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
        self.name = '%d+%s'%(ith_player, name)
        self.cost = cost
        self.player = ith_player
        super(DecisionNode, self).register_parent(upper, ith_child)
        for i in range(N_PLAYERS):
            if i == self.player:
                self.visible[i] = 1
            else:
                self.visible[i] = 0
        self.n_children = n_children
        feature = [f for (v,f) in zip(self.visible, self.feature) if v]
        self.policy_maker = PolicyMaker(n_children, feature, self)
        self.policy = [mx.sym.Variable('%s_%d_policy'%(self.name, i)) for i in range(n_children)]
        super(DecisionNode, self).alloc_children()
        for i in range(self.n_children):
            self.pout[i] = self.policy[i] * self.pin
        self.arg_names = ['%s_%d_policy'%(self.name, i) for i in range(n_children)]
        self.arg_nds = [mx.nd.zeros([BATCH_SIZE], CTX) for i in range(n_children)]
        self.arg_dict = dict([(name, nd) for name,nd in zip(self.arg_names, self.arg_nds)])


class RevealNode(Node):
    def __init__(self, upper, ith_child, new_visible, name):
        self.name = name
        self.cost = [0] * N_PLAYERS
        self.n_children = 1
        super(RevealNode, self).register_parent(upper, ith_child)
        for d in new_visible:
            self.visible[d] = 1
        super(RevealNode, self).alloc_children()
        self.pout[0] = self.pin


class AwardNode(Node):
    def __init__(self, upper, ith_child, winner, name):
        self.name = name
        super(AwardNode, self).register_parent(upper, ith_child)
        self.loss[winner] -= self.pot
        self.pout = [self.pin]
        self.grad = mx.nd.zeros([BATCH_SIZE], CTX)

    def get_grad(self, player):
        self.grad[:] = self.loss[player]



class ShowdownNode(Node):
    def __init__(self, upper, ith_child, name):
        self.name = name
        super(ShowdownNode, self).register_parent(upper, ith_child)
        self.loss = np.expand_dims(np.array(self.loss), 0)
        self.loss = np.tile(self.loss, [BATCH_SIZE, 1])
        self.share = np.zeros([BATCH_SIZE, N_PLAYERS])
        self.pout = [self.pin]
        self.grad = mx.nd.zeros([BATCH_SIZE], CTX)

    def get_grad(self, player):
        self.grad[:] = self.loss[:,player] - self.share[:,player] * self.pot


def train():
    root = RootNode('root')

    pf1d = DecisionNode(root, 0, 0, 3, [2,1,0], 'pf1d')
    flrv1 = RevealNode(pf1d, 1, [2], 'flrv1')
    fl1d1 = DecisionNode(flrv1, 0, 0, 3, [1,0,0], 'fl1d1')
    fl2f1 = AwardNode(fl1d1, 2, 1, 'fl2f1')
    fl2s1 = ShowdownNode(fl1d1, 1, 'fl2s1')
    fl2d1 = DecisionNode(fl1d1, 0, 1, 3, [2,1,0], 'fl2d1')
    fl3f1 = AwardNode(fl2d1, 2, 0, 'fl3f1')
    fl3s1 = ShowdownNode(fl2d1, 1, 'fl3s1')
    fl3d1 = DecisionNode(fl2d1, 0, 0, 3, [2,1,0], 'fl3d1')
    fl4f1 = AwardNode(fl3d1, 2, 1, 'fl4f1')
    fl4s1 = ShowdownNode(fl3d1, 1, 'fl4s1')
    fl4d1 = DecisionNode(fl3d1, 0, 1, 2, [1,0], 'fl4d1')
    fl5f1 = AwardNode(fl4d1, 1, 0, 'fl5f1')
    fl5s1 = ShowdownNode(fl4d1, 0, 'fl5s1')
    pf2f = AwardNode(pf1d, 2, 1, 'pf2f')

    pf2d = DecisionNode(pf1d, 0, 1, 3, [2,1,0], 'pf2d')
    flrv2 = RevealNode(pf2d, 1, [2], 'flrv2')
    fl1d2 = DecisionNode(flrv2, 0, 0, 3, [1,0,0], 'fl1d2')
    fl2f2 = AwardNode(fl1d2, 2, 1, 'fl2f2')
    fl2s2 = ShowdownNode(fl1d2, 1, 'fl2s2')
    fl2d2 = DecisionNode(fl1d2, 0, 1, 3, [2,1,0], 'fl2d2')
    fl3f2 = AwardNode(fl2d2, 2, 0, 'fl3f2')
    fl3s2 = ShowdownNode(fl2d2, 1, 'fl3s2')
    fl3d2 = DecisionNode(fl2d2, 0, 0, 3, [2,1,0], 'fl3d2')
    fl4f2 = AwardNode(fl3d2, 2, 1, 'fl4f2')
    fl4s2 = ShowdownNode(fl3d2, 1, 'fl4s2')
    fl4d2 = DecisionNode(fl3d2, 0, 1, 2, [1,0], 'fl4d2')
    fl5f2 = AwardNode(fl4d2, 1, 0, 'fl5f2')
    fl5s2 = ShowdownNode(fl4d2, 0, 'fl5s2')
    pf3f = AwardNode(pf2d, 2, 0, 'pf3f')

    pf3d = DecisionNode(pf2d, 0, 0, 2, [1,0], 'pf3d')
    flrv3 = RevealNode(pf3d, 0, [2], 'flrv3')
    fl1d3 = DecisionNode(flrv3, 0, 0, 3, [1,0,0], 'fl1d3')
    fl2f3 = AwardNode(fl1d3, 2, 1, 'fl2f3')
    fl2s3 = ShowdownNode(fl1d3, 1, 'fl2s3')
    fl2d3 = DecisionNode(fl1d3, 0, 1, 3, [2,1,0], 'fl2d3')
    fl3f3 = AwardNode(fl2d3, 2, 0, 'fl3f3')
    fl3s3 = ShowdownNode(fl2d3, 1, 'fl3s3')
    fl3d3 = DecisionNode(fl2d3, 0, 0, 3, [2,1,0], 'fl3d3')
    fl4f3 = AwardNode(fl3d3, 2, 1, 'fl4f3')
    fl4s3 = ShowdownNode(fl3d3, 1, 'fl4s3')
    fl4d3 = DecisionNode(fl3d3, 0, 1, 2, [1,0], 'fl4d3')
    fl5f3 = AwardNode(fl4d3, 1, 0, 'fl5f3')
    fl5s3 = ShowdownNode(fl4d3, 0, 'fl5s3')
    pf4f = AwardNode(pf3d, 1, 1, 'pf4f')

    leaf_nodes = [pf2f, pf3f, pf4f,
                    fl2s1, fl2f1, fl3s1, fl3f1, fl4s1, fl4f1, fl5s1, fl5f1,
                    fl2s2, fl2f2, fl3s2, fl3f2, fl4s2, fl4f2, fl5s2, fl5f2,
                    fl2s3, fl2f3, fl3s3, fl3f3, fl4s3, fl4f3, fl5s3, fl5f3
                    ]
    showdown_nodes = [fl2s1, fl3s1, fl4s1, fl5s1,
                    fl2s2, fl3s2, fl4s2, fl5s2,
                    fl2s3, fl3s3, fl4s3, fl5s3
                    ]
                    
    param_nodes = [pf1d, pf2d, pf3d,
                    fl1d1, fl2d1, fl3d1, fl4d1,
                    fl1d2, fl2d2, fl3d2, fl4d2,
                    fl1d3, fl2d3, fl3d3, fl4d3
                    ]
    outs = map(lambda x:x.pout[0], leaf_nodes)
    args = {}
    args.update(root.arg_dict)
    for node in param_nodes:
        args.update(node.arg_dict)
    grad_dict = {}
    for k in args:
        if not k.startswith('player') and not k.startswith('board') and not k.startswith('root'):
            grad_dict[k] = mx.nd.zeros(args[k].shape, CTX)

    final_out = mx.sym.Group(outs)
    graph = final_out.bind(ctx=CTX, args=args, args_grad=grad_dict, grad_req='write')
    for dnode in param_nodes:
        dnode.policy_maker.get_executor(root.arg_dict)

    total_loss = np.zeros([13,13])
    total_count = np.zeros([13,13])

    count = 0
    while True:
        count += 1
        win_shares, cc0, cc1, cc2 = root.sample()
        for node in showdown_nodes:
            node.share[:] = win_shares
        for j in range(N_PLAYERS):
            for dnode in param_nodes:
                dnode.policy_maker.forward(graph)
            graph.forward(is_train=True)
            for node in leaf_nodes:
                node.get_grad(j)
            if j == 0:
                xx = np.zeros([BATCH_SIZE]) 
                for i, node in enumerate(leaf_nodes):
                    if count % 100 == 1 and graph.outputs[i].asnumpy().mean() > 0.1:
                        print node.name, graph.outputs[i].asnumpy().mean()
                    xx += (graph.outputs[i] * node.grad).asnumpy()
            graph.backward([-node.grad for node in leaf_nodes])
            for dnode in param_nodes:
                if dnode.name[0] == str(j):
                    dnode.policy_maker.update([graph.grad_dict['%s_%d_policy'%(dnode.name, i)] for i in range(dnode.n_children)])
        for i in range(BATCH_SIZE):
            total_loss[cc0[i], cc1[i]] += xx[i]
            total_count[cc0[i], cc1[i]] += 1
        if count % 100 == 1:
            for i in range(13):
                for j in range(13):
                    print "%.4f\t" % (-total_loss[i,j]/total_count[i,j]),
                print
            print '\n'
        if count % 1000 == 0:
            total_loss *= 0
            total_count *= 0
            for dnode in param_nodes:
                dnode.policy_maker.dump('model')

def test():
    root = RootNode('root')

    pf1d = DecisionNode(root, 0, 0, 3, [2,1,0], 'pf1d')
    flrv1 = RevealNode(pf1d, 1, [2], 'flrv1')
    fl1d1 = DecisionNode(flrv1, 0, 0, 3, [1,0,0], 'fl1d1')
    fl2f1 = AwardNode(fl1d1, 2, 1, 'fl2f1')
    fl2s1 = ShowdownNode(fl1d1, 1, 'fl2s1')
    fl2d1 = DecisionNode(fl1d1, 0, 1, 3, [2,1,0], 'fl2d1')
    fl3f1 = AwardNode(fl2d1, 2, 0, 'fl3f1')
    fl3s1 = ShowdownNode(fl2d1, 1, 'fl3s1')
    fl3d1 = DecisionNode(fl2d1, 0, 0, 3, [2,1,0], 'fl3d1')
    fl4f1 = AwardNode(fl3d1, 2, 1, 'fl4f1')
    fl4s1 = ShowdownNode(fl3d1, 1, 'fl4s1')
    fl4d1 = DecisionNode(fl3d1, 0, 1, 2, [1,0], 'fl4d1')
    fl5f1 = AwardNode(fl4d1, 1, 0, 'fl5f1')
    fl5s1 = ShowdownNode(fl4d1, 0, 'fl5s1')
    pf2f = AwardNode(pf1d, 2, 1, 'pf2f')

    pf2d = DecisionNode(pf1d, 0, 1, 3, [2,1,0], 'pf2d')
    flrv2 = RevealNode(pf2d, 1, [2], 'flrv2')
    fl1d2 = DecisionNode(flrv2, 0, 0, 3, [1,0,0], 'fl1d2')
    fl2f2 = AwardNode(fl1d2, 2, 1, 'fl2f2')
    fl2s2 = ShowdownNode(fl1d2, 1, 'fl2s2')
    fl2d2 = DecisionNode(fl1d2, 0, 1, 3, [2,1,0], 'fl2d2')
    fl3f2 = AwardNode(fl2d2, 2, 0, 'fl3f2')
    fl3s2 = ShowdownNode(fl2d2, 1, 'fl3s2')
    fl3d2 = DecisionNode(fl2d2, 0, 0, 3, [2,1,0], 'fl3d2')
    fl4f2 = AwardNode(fl3d2, 2, 1, 'fl4f2')
    fl4s2 = ShowdownNode(fl3d2, 1, 'fl4s2')
    fl4d2 = DecisionNode(fl3d2, 0, 1, 2, [1,0], 'fl4d2')
    fl5f2 = AwardNode(fl4d2, 1, 0, 'fl5f2')
    fl5s2 = ShowdownNode(fl4d2, 0, 'fl5s2')
    pf3f = AwardNode(pf2d, 2, 0, 'pf3f')

    pf3d = DecisionNode(pf2d, 0, 0, 2, [1,0], 'pf3d')
    flrv3 = RevealNode(pf3d, 0, [2], 'flrv3')
    fl1d3 = DecisionNode(flrv3, 0, 0, 3, [1,0,0], 'fl1d3')
    fl2f3 = AwardNode(fl1d3, 2, 1, 'fl2f3')
    fl2s3 = ShowdownNode(fl1d3, 1, 'fl2s3')
    fl2d3 = DecisionNode(fl1d3, 0, 1, 3, [2,1,0], 'fl2d3')
    fl3f3 = AwardNode(fl2d3, 2, 0, 'fl3f3')
    fl3s3 = ShowdownNode(fl2d3, 1, 'fl3s3')
    fl3d3 = DecisionNode(fl2d3, 0, 0, 3, [2,1,0], 'fl3d3')
    fl4f3 = AwardNode(fl3d3, 2, 1, 'fl4f3')
    fl4s3 = ShowdownNode(fl3d3, 1, 'fl4s3')
    fl4d3 = DecisionNode(fl3d3, 0, 1, 2, [1,0], 'fl4d3')
    fl5f3 = AwardNode(fl4d3, 1, 0, 'fl5f3')
    fl5s3 = ShowdownNode(fl4d3, 0, 'fl5s3')
    pf4f = AwardNode(pf3d, 1, 1, 'pf4f')

    leaf_nodes = [pf2f, pf3f, pf4f,
                    fl2s1, fl2f1, fl3s1, fl3f1, fl4s1, fl4f1, fl5s1, fl5f1,
                    fl2s2, fl2f2, fl3s2, fl3f2, fl4s2, fl4f2, fl5s2, fl5f2,
                    fl2s3, fl2f3, fl3s3, fl3f3, fl4s3, fl4f3, fl5s3, fl5f3
                    ]
    showdown_nodes = [fl2s1, fl3s1, fl4s1, fl5s1,
                    fl2s2, fl3s2, fl4s2, fl5s2,
                    fl2s3, fl3s3, fl4s3, fl5s3
                    ]
    param_nodes = [pf1d, pf2d, pf3d,
                    fl1d1, fl2d1, fl3d1, fl4d1,
                    fl1d2, fl2d2, fl3d2, fl4d2,
                    fl1d3, fl2d3, fl3d3, fl4d3
                    ]
    outs = map(lambda x:x.pout[0], leaf_nodes)
    args = {}
    args.update(root.arg_dict)
    for node in param_nodes:
        args.update(node.arg_dict)
    grad_dict = {}
    for k in args:
        if not k.startswith('player') and not k.startswith('board') and not k.startswith('root'):
            grad_dict[k] = mx.nd.zeros(args[k].shape, CTX)

    final_out = mx.sym.Group(outs)
    graph = final_out.bind(ctx=CTX, args=args, args_grad=grad_dict, grad_req='write')
    for dnode in param_nodes:
        dnode.policy_maker.get_executor(root.arg_dict)
    for dnode in param_nodes:
        dnode.policy_maker.load('model')

    while True:
        win_shares = root.manual()
        for node in showdown_nodes:
            node.share[:] = win_shares
        for j in range(N_PLAYERS):
            for node in leaf_nodes:
                node.get_grad(j)
            for dnode in param_nodes:
                dnode.policy_maker.forward(graph)
                print dnode.name, [graph.arg_dict['%s_%d_policy'%(dnode.name, i)].asnumpy()[0] for i in range(dnode.n_children)]
            graph.forward(is_train=True)
            graph.backward([-node.grad for node in leaf_nodes])
            for dnode in param_nodes:
                if dnode.name[0] == str(j):
                    print j, dnode.name, [graph.grad_dict['%s_%d_policy'%(dnode.name, i)].asnumpy()[0]\
                            / max(1e-30, np.sum(np.abs(graph.grad_dict['%s_%d_policy'%(dnode.name,i)].asnumpy()[0]))) for i in range(dnode.n_children)]


if __name__ == '__main__':
    train()
