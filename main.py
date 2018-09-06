import numpy as np
from numpy import linalg as LA
from scipy import signal

import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

FLOOR = 0
ROCK = 1
WALL = 2

C = 50
R = .50
N = 4
T = 5
M = 1

MAKE_WALLS = True
def init_grid(c,r):
    """ Initializes a cxc grid of cells where r is the ratio of nonzero
    cells, i.e. r = (# nonzero cells)/(cxc),
    """
    num_nonzero = int(np.floor(r*(c*c)))

    # nifty function which initializes a vector of size num_nonzero
    # sampling from a uniform distribution over 0 to c*c without
    # replacement.
    x_nz = np.random.choice(c*c,num_nonzero,replace=False)
    grid = np.zeros(c*c)
    for i in x_nz:
        grid[i]=ROCK
    return grid.reshape((c,c))


def display_grid(grid):
    fig, ax = plt.subplots()

    norm_grid = grid/2
    im = ax.pcolormesh(norm_grid,cmap='RdBu')
    plt.colorbar(im, ax=ax)
    plt.show()


def get_wall_mask():
    mask = np.ones((3,3))
    mask[0,0] = 0
    mask[0,2] = 0
    mask[2,0] = 0
    mask[2,2] = 0
    mask[1,1] = 0
    return mask


def define_walls(grid):
    mask = get_wall_mask()
    grad = signal.convolve2d(grid,mask,mode='same')
    change_idxs = np.where(np.logical_and(np.greater_equal(grad,1), np.less(grad, 4)))
    for x,y in zip(change_idxs[0],change_idxs[1]):
        if grid[x,y] == ROCK:
            grid[x,y]=WALL
    return grid


def ca_iterate(grid,T,M):
    ''' Performs an iteration of the cellular automata algorithm.'''
    dim = 2*M+1
    mask = np.ones((dim,dim))
    center = int(np.floor(dim/2))
    mask[center,center] = 0
    print(mask)
    grad = signal.convolve2d(grid,mask,mode='same')
    change_idxs = np.where(grad>=T)
    grid = np.ones((C,C))
    for x,y in zip(change_idxs[0],change_idxs[1]):
        grid[x,y]=FLOOR
    return grid

if __name__=="__main__":
    grid = init_grid(C,R)

    for _ in range(N):
        last_grid = np.copy(grid)
        grid = ca_iterate(grid,T,M)
    if MAKE_WALLS:
        grid = define_walls(grid)
    display_grid(grid)

