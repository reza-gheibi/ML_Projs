# Both the members in our group are beginners in Python
# which is reflected in the way we wrote the code, apologies for that

# Towers of Hanoi - 3 disks - Double Q-learning

import numpy as np

def toh():

    num_states = 27 # total number of states 27 (0-26) for 3 disks
    init_state = 0  # 0 is the initial state
    fin_state = 26  # 26 is the final state
    
    # defining R matrix
    R = [[float('-inf') for column in range(num_states)] for row in range(num_states)]
    #all possible actions have reward of -0.01 except actions to goal state which have reward of 1
    R[0][1]=R[0][2]=-.01
    R[1][0]=R[1][2]=R[1][3]=-.01
    R[2][0]=R[2][1]=R[2][4]=-.01
    R[3][1]=R[3][5]=R[3][6]=-.01
    R[4][2]=R[4][7]=R[4][8]=-.01
    R[5][3]=R[5][6]=R[5][9]=-.01
    R[6][3]=R[6][5]=R[6][7]=-.01
    R[7][4]=R[7][6]=R[7][8]=-.01
    R[8][4]=R[8][7]=R[8][10]=-.01
    R[9][5]=R[9][11]=R[9][12]=-.01
    R[10][8]=R[10][13]=R[10][14]=-.01
    R[11][9]=R[11][12]=R[11][15]=-.01
    R[12][9]=R[12][11]=R[12][16]=-.01
    R[13][10]=R[13][14]=R[13][17]=-.01
    R[14][10]=R[14][13]=R[14][18]=-.01
    R[15][11]=R[15][19]=R[15][20]=-.01
    R[16][12]=R[16][21]=R[16][22]=-.01
    R[17][13]=R[17][23]=R[17][24]=-.01
    R[18][14]=R[18][25]=-.01
    R[19][15]=R[19][20]=-.01
    R[20][15]=R[20][19]=R[20][21]=-.01
    R[21][16]=R[21][20]=R[21][22]=-.01
    R[22][16]=R[22][21]=R[22][23]=-.01
    R[23][17]=R[23][22]=R[23][24]=-.01
    R[24][17]=R[24][23]=R[24][25]=-.01
    R[25][18]=R[25][24]=-.01
    R[26][18]=R[26][25]=-.01
    R[18][26]=R[25][26]=1

    # initializing Q1 and Q2 matrices with all 0s
    Q1 = [[0 for column in range(num_states)] for row in range(num_states)]
    Q2 = [[0 for column in range(num_states)] for row in range(num_states)]

    # learning parameters    
    gamma = 0.9     # gamma value
    alpha = 1       # alpha value
    episodes = 10000    # number of training episodes

    # loop for episodes
    for x in range(episodes):

        # choosing a random initial state
        current_state = np.random.randint(0,num_states)

        # loop until final state is reached
        while current_state != fin_state:
            
            # finding possible actions first
            j = 0
            actions = [0] * 3
            for i in range(num_states):
                if R[current_state][i] != float('-inf'):
                    actions[j] = i
                    j += 1
                    
            # choosing action following epsilon-greedy policy in Q1 + Q2
            if np.random.uniform() <= 0.05:
                next_state = actions[np.random.randint(0,j)]
                
            else: # greedy policy
                find_max = [-100] * j
                for t in range(j):
                    find_max[t] = Q1[current_state][actions[t]] + Q2[current_state][actions[t]]
                max_qa = max(find_max)
                
                j1 = 0
                actions1 = [0] * 3
                for i in actions[0:j]:
                    if Q1[current_state][i] + Q2[current_state][i] == max_qa:
                        actions1[j1] = i
                        j1 += 1
                next_state = actions1[np.random.randint(0,j1)]

            # Updating Q1, Q2 with 0.5 probability
            j = 0
            actions = [0] * 3
            find_max = [-100] * 3
            if np.random.randint(0,2) < 1:
                for t in range(num_states):
                    if R[next_state][t] != float('-inf'):
                        find_max[j] = Q1[next_state][t]
                        j += 1
                m = max(find_max)
                j = 0
                for i in range(num_states):
                    if Q1[next_state][i] == m and R[next_state][i] != float('-inf'):
                        actions[j] = i
                        j += 1
                next_state1 = actions[np.random.randint(0,j)]
                max_q = Q2[next_state][next_state1]
                Q1[current_state][next_state]=Q1[current_state][next_state]+alpha*(R[current_state][next_state]+gamma*max_q-Q1[current_state][next_state])
                
            else:
                j = 0 
                for t in range(num_states):
                    if R[next_state][t] != float('-inf'):
                        find_max[j] = Q2[next_state][t]
                        j += 1
                m = max(find_max)
                j = 0
                for i in range(num_states):
                    if Q2[next_state][i] == m and R[next_state][i] != float('-inf'):
                        actions[j] = i
                        j += 1
                next_state1 = actions[np.random.randint(0,j)]
                max_q = Q1[next_state][next_state1]
                Q2[current_state][next_state]=Q2[current_state][next_state]+alpha*(R[current_state][next_state]+gamma*max_q-Q2[current_state][next_state])
                
            current_state = next_state

    
    # now agent is solving the puzzle
    state = init_state  # starts at initial state
    num_moves = 0       # count for number of moves used
    reward = 0          # accumulator for reward

    # loop until final state is reached
    while state != fin_state:
        j = 0
        find_max = [-100] * 3
        for t in range(num_states):
            if R[state][t]!= float('-inf'):
                find_max[j] = Q1[state][t]
                j += 1
        max_qa = max(find_max)
        j1 = 0
        max_actions = [0] * 3
        for i in range(num_states):
            if Q1[state][i] == max_qa and R[state][i]!= float('-inf'):
                max_actions[j1] = i
                j1 += 1

        # taking action, updating number of moves and reward
        next_state = max_actions[np.random.randint(0,j1)]
        num_moves += 1
        reward += R[state][next_state]
        state = next_state
        
    print("Number of training episodes: "+ str(episodes))
    print("Number of moves: " + str(num_moves))
    print("Reward for the agent: " + str(reward))

toh()
