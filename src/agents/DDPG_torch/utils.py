from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def log_plot(env, agent, cube, slice = 30, wandb = True):
    clear_output(True)
    state, done = env.reset()
    traj_x = [state[0]]
    traj_y = [state[1]]
    traj_z = [state[2]]
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        traj_x.append(state[0])
        traj_y.append(state[1])
        traj_z.append(state[2])

    if wandb:
        plt.plot(traj_z, traj_y, c='r')
        wandb.log({"chart": plt})
    else:
        plt.plot(traj_z, traj_y, c='r')
        plt.imshow(cube[slice, :, :], cmap='viridis')



