import gym
import random
import settings
import numpy as np
from deap import gp


def softmax(*args):
    values = [arg for arg in args]
    e_x = np.exp(np.array(values) - np.max(np.array(values)))
    values = e_x / e_x.sum()
    values = values.tolist()[0]
    max_value = max(values)

    return values.index(max_value)


def mate(ind_1, ind_2, creator):
    ind_1_list = [ind for ind in ind_1]
    ind_2_list = [ind for ind in ind_2]
    n_1 = []
    n_2 = []
    for i_1, i_2 in zip(ind_1_list, ind_2_list):
        new_1, new_2 = gp.cxOnePoint(i_1, i_2)
        n_1.append(new_1)
        n_2.append(new_2)

    return creator.Individual(n_1), creator.Individual(n_2)


def create_attribute(pset):
    tree =  gp.genGrow(pset=pset, 
                      min_=settings.MINIMAL_TREE_DEPTH,
                      max_=settings.MAXIMUM_TREE_DEPTH)

    return gp.PrimitiveTree(tree)


def mutate(ind, pset):
    index = random.randint(0, settings.OUTPUT_NUMBER-1)
    ind[index] = create_attribute(pset)

    return ind,


def get_individual_output(individual, input_data):
    return softmax([f(*input_data) for f in individual])


def evaluate_individual(individual, toolbox, render=False):
    env = gym.make(settings.ENV_NAME)
    observation = env.reset()
    total_reward = 0
    individual = [toolbox.compile(expr=ind) for ind in individual] 

    for _ in range(settings.NUMBER_OF_STEPS):
        if render:
            env.render()
        action = get_individual_output(individual=individual,
                                       input_data=observation)

        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(total_reward)
            return (total_reward,)

    env.close()
    print(total_reward)
    return (total_reward,)


