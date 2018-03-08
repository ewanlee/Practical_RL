from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from estool.es import PEPG
from tqdm import tqdm
import coloredlogs
import logging
from pong import make_pong
import numpy as np
import os

def create_model(env):
    """
    Create policy network.

    Inputs:
    - env (gym.env): the gym environment object

    Outputs:
    - model (keras.Sequential): the policy network
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu',
                     data_format='channels_first', input_shape=(4, 42, 42)))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(env.action_space.n, activation='softmax'))

    return model

def get_param_shapes(model):
    """
    Get the weight shape of each layer of the policy network

    Inputs:
    - model (keras.Sequential): the policy network

    Outputs:
    - param_shapes (List): each element is a multi-dimensional tuple that represents each layer's weight shape
    """
    param_shapes = []
    for weight in model.get_weights():
        param_shapes.append(np.shape(weight))
    return param_shapes

def set_params(model, param, param_shape_list):
    """
    Set the weights of each layer of the policy network

    Inputs:
    - model (keras.Sequential): the policy network
    - param (List): the list of all weights of the policy network
    - param_shape_list (List): each element is a multi-dimensional tuple that represents each layer's weight shape

    Outputs:
    - model (keras.Sequential): the policy network
    """
    param_nums = [int(np.prod(shape)) for shape in param_shape_list]
    params = []
    pos = 0
    for ind, num in enumerate(param_nums):
        params.append(np.reshape(param[pos:pos+num],
                                 param_shape_list[ind]))
        pos += num
    model.set_weights(params)
    return model


def fit_func(model, env, t_max=10**4):
    """
    The fitness function of the evolution strategy algorithm

    Inputs:
    - model (keras.Sequential): the policy network
    - env (gym.env): the gym environment object
    - tmax (integer): the maximum steps that agent will take

    Outputs:
    - total_reward (float): the total reward the agent get
    """
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        flag = np.random.choice(range(0, 2), p=[0.1, 0.9])
        if flag == 1:
            a = np.argmax(model.predict(np.expand_dims(s, 0))[0])
        else:
            a = np.random.choice(range(env.action_space.n), p=np.ones((env.action_space.n, ))*(1 / env.action_space.n))
        new_s, r, done, _ = env.step(a)
        total_reward += r

        s = new_s
        if done:
            break
    return total_reward

def es(solver, model, env, param_shape_list):
    history = []
    for j in range(4000):
        solutions = solver.ask()
        fitness_list = [fit_func(set_params(
            model, solutions[i], param_shape_list), env) for i in tqdm(range(solver.popsize))]
        solver.tell(fitness_list)
        best_sol, global_best_fitness, best_fitness, _  = solver.result()
        history.append(best_fitness)
        logger.debug("fitness at iteration {}: {}".format((j+1), best_fitness))
    logger.debug("fitness score at this local optimum: {}".format(global_best_fitness))

def main():
    env = make_pong()
    model = create_model(env)
    param_shape_list = get_param_shapes(model)
    NPARAMS = np.sum([int(np.prod(shape)) for shape in param_shape_list])
    # defines PEPG (NES) solver
    pepg = PEPG(NPARAMS,           # number of model parameters
                sigma_init=0.5,                  # initial standard deviation
                learning_rate=0.1,               # learning rate for standard deviation
                learning_rate_decay=1.0,       # don't anneal the learning rate
                popsize=100,             # population size
                average_baseline=True,          # set baseline to average of batch
                weight_decay=0.00,            # weight decay coefficient
                rank_fitness=False,           # use rank rather than fitness numbers
                forget_best=False)            # don't keep the historical best solution)
    es(pepg, model, env, param_shape_list)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    formats = '[%(asctime)-15s %(pathname)s:%(lineno)-3s] %(message)s'
    logger = logging.getLogger('ae_pred')

    coloredlogs.install(mt=formats, level='DEBUG', logger=logger)

    main()
