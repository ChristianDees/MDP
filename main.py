#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Christian Dees & Aitiana Mondragon
"""

import argparse
import random


# States & actions
states = ['RU8p', 'TU10p', 'RU10p', 'RD10p', 'RU8a', 'RD8a', 'TU10a', 'RU10a', 'RD10a', 'TD10a', 'terminal']
actions = ['P', 'R', 'S', 'any']

# (state, action) -> [(probability, reward, next_state)]
transitions = {
    ('RU8p', 'P'): [(1.0, 2, 'TU10p')],
    ('RU8p', 'R'): [(1.0, 0, 'RU10p')],
    ('RU8p', 'S'): [(1.0, -1, 'RD10p')],
    ('TU10p', 'P'): [(1.0, 2, 'TU10a')],
    ('TU10p', 'R'): [(1.0, 0, 'RU8a')],
    ('RU10p', 'P'): [(0.5, 2, 'RU8a'), (0.5, 2, 'TU10a')],
    ('RU10p', 'R'): [(1.0, 0, 'RU8a')],
    ('RU10p', 'S'): [(1.0, -1, 'RD8a')],
    ('RD10p', 'P'): [(0.5, 2, 'RD8a'), (0.5, 2, 'TD10a')],
    ('RD10p', 'R'): [(1.0, 0, 'RD8a')],
    ('RU8a', 'P'): [(1.0, 2, 'RU10a')],
    ('RU8a', 'R'): [(1.0, 0, 'RU10a')],
    ('RU8a', 'S'): [(1.0, -1, 'RD10a')],
    ('RD8a', 'P'): [(1.0, 2, 'RD10a')],
    ('RD8a', 'R'): [(1.0, 0, 'RD10a')],
    ('TU10a', 'any'): [(1.0, -1, 'terminal')],
    ('RU10a', 'any'): [(1.0, 0, 'terminal')],
    ('RD10a', 'any'): [(1.0, 4, 'terminal')],
    ('TD10a', 'any'): [(1.0, 3, 'terminal')]
}


# Value Iteration Algorithm 
def valueIteration():
    gamma = 0.99   # Discount factor 
    theta = 0.001  # Convergence threshold
    
    # Init all state values to 0.0 
    ve = {state: 0.0 for state in states}  
    policy = {state: None for state in states} # Best action for each state
    
    # Total iterations
    iteration = 0  
    delta = float('inf') 
    
    # Iterates until the change in values is small enough
    while delta > theta:
        iteration += 1 
        delta = 0 
    
        # Update each state value estimate
        for state in states:
            actionVals = {} 
            preVe = ve[state] # Prev value
            # Actions for current state
            for action in actions:
                if (state, action) in transitions:
                    qVal = 0  
                    # Possible transitions for current state
                    for prob, reward, nextState in transitions[(state, action)]:
                        # Terminal -> reward
                        # Else -> reward plus discounted value of the next state
                        if nextState == 'terminal': qVal += prob * reward
                        else: qVal += prob * (reward + gamma * ve[nextState])
    
                    # Update qval for this action
                    actionVals[action] = qVal
    
            # Skip to next state if no valid actions
            if not actionVals: continue
    
            # Get best action & val for current state
            maxAction = max(actionVals, key=actionVals.get)  
            maxVal = actionVals[maxAction] 
    
            # Display current state info
            print(f"State {state}:")
            print("-"*20)
            print(f"Previous Value: {ve[state]}")  
            print(f"New Value: {maxVal}")  
            print(f"Action Values: {actionVals}")  
            print(f"Best Action: {maxAction}\n") 
    
            # Update delta, value estimate, and policy
            delta = max(delta, abs(preVe - maxVal))
            ve[state] = maxVal
            policy[state] = maxAction

    # Display results after convergence
    print("="*40)
    print("Value Iteration Converged!".center(40))
    print("="*40)
    
    # Display total iterations & state vals
    print(f"Total Iterations: {iteration}\n") 
    print("Final State Values:")
    print("-"*40)
    for state in states: print(f"{state}: {ve[state]:.4f}")  
    
    # Display best action for each state
    print("\nOptimal Policy:")
    print("-"*40)
    for state in states: print(f"{state}: {policy[state]}")
    print("="*40)
    
    
# Q-Learning Algorithm
def QLearning():
    # PARAMETERS
    end_flag = 0.001  # max change in q-value => stop episodes if met
    alpha = 0.1       # learning rate
    mylambda = 0.99   # discount rate
    exploit_rate = 0.80  # optimal action
    explore_rate = 0.20  # random action => epsilon value
    q_table = {}         # q table
    max_change = float('inf')

    # initialize q table
    for state in states:
        q_table[state] = {}
        for action in actions:
            q_table[state][action] = 0.0

    # going through episodes
    episode = 0  # tracking iterations
    while max_change > end_flag:
        delta = 0
        episode += 1
        current_state = random.choice([s for s in states if s != 'terminal'])

        while current_state != 'terminal':
            available_actions = [a for a in actions if (current_state, a) in transitions and a != 'any']
            if not available_actions:
                break

            # ε-greedy policy
            if random.random() < explore_rate:
                action = random.choice(available_actions)  # explore
            else:
                action = max(available_actions, key=lambda a: q_table[current_state][a])  # exploit

            # Retrieve transition (assumes deterministic or first outcome only)
            probability, reward, next_state = transitions[(current_state, action)][0]

            old_q = q_table[current_state][action]

            # Q-learning update rule
            if next_state == 'terminal':
                q_next = 0  # no future rewards
                new_q = old_q + alpha * (reward - old_q)
            else:
                q_next = max(q_table[next_state].values()) if q_table[next_state] else 0
                new_q = old_q + alpha * (reward + mylambda * q_next - old_q)

            q_table[current_state][action] = new_q

            # Print required debug output
            print(f"Updating Q[{current_state}][{action}]")
            print(f"  Previous Q-value: {old_q:.4f}")
            print(f"  New Q-value:      {new_q:.4f}")
            print(f"  Reward:           {reward}")
            print(f"  Max Q(next state):{q_next:.4f}\n")

            delta = max(delta, abs(new_q - old_q))
            current_state = next_state

        max_change = delta

    # After learning, print summary
    print("=" * 40)
    print("Q-Learning Converged!".center(40))
    print("=" * 40)
    print(f"  Total Episodes: {episode}\n")

    print("\n  Learned Q-table:")
    print("-" * 40)
    for state in states:
        print(f"    {state}:")
        for action, value in q_table[state].items():
            print(f"      {action}: {value:.4f}")
    print("-" * 40)

    # Derive optimal policy
    policy_q = {}
    for state in states:
        if state != 'terminal':
            available_actions = [a for a in actions if a in q_table[state] and (state, a) in transitions and a != 'any']
            if available_actions:
                policy_q[state] = max(available_actions, key=lambda a: q_table[state][a])
            else:
                policy_q[state] = None
        else:
            policy_q[state] = None

    print("\n  Optimal Policy (from Q-Learning):")
    print("-" * 40)
    for state, action in policy_q.items():
        print(f"    {state}: {action}")
    print("-" * 40)

    return q_table, policy_q

                    

    
def main():
    parser = argparse.ArgumentParser(description="Solving a Markov Decision Process.")
    parser.add_argument('-v', '--value-iteration', action='store_true', help="Use value iteration algorithm")
    parser.add_argument('-q', '--Q-Learning', action='store_true', help="Use Q-Learning algorithm")
    args = parser.parse_args()
    if args.value_iteration:valueIteration()
    elif args.Q_Learning: QLearning()
    else:print("Please select an algorithm to run.\nUse '-h' or '--help' for a list of available options.")


if __name__ == "__main__":
    main()