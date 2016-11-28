import numpy as np
import time

# ZeroSum Game: player1 and player2 each dealt a card from 1~9
# player1 has two moves: move0 award -1, move1 let player2 decide.
# player2 move0 award 1, move1 show down and winner get 2, loser get -2.

p = np.ones(9) * 0.5 # initialize player1's policy
q = np.ones(9) * 0.5 # player2's policy
lr = 1e-1

compare_table = np.ones([9,9]) * -1
for i in range(9):
    for j in range(i, 9):
        compare_table[j,i] = 1
compare_table[0,8] = 1
compare_table[8,0] = -1

for iter in range(1000):
    if iter % 100 == 0:
        lr /= 2 
    # update q
    for i in range(9):
        e0 = 0 
        e1 = 0
        for j in range(9):
            if j == i:
                continue
            e1 += compare_table[i,j] * p[j]
            e1 += 0 * (1-p[j])
        q[i] += lr * (e1-e0) 
    q[q>1] = 1
    q[q<0] = 0

    # update p
    x = []
    for j in range(9):
        total_q = np.sum(q) - q[j]
        e0 = 0 
        e1 = 0
        for i in range(9):
            if i == j:
                continue
            e1 += compare_table[j,i] * q[i]
            e1 += 0 * (1-p[i])
        p[j] += lr * (e1-e0)
    p[p>1] = 1
    p[p<0] = 0

print map(lambda x:round(x,2), p.tolist()), map(lambda x:round(x,2), q.tolist())
