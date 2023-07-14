import gym
from gym import spaces
import numpy as np
import pybullet as p
class ReacherEnv(gym.Env):
    """Custom environment to train an agent to perform inverse kinematics"""

    def __init__(self,robot):
        super(ReacherEnv, self).__init__()
        print('Environment initialized')

        self.robot = robot
        self.num_joints = p.getNumJoints(robot)
        self.t = 0
        self.target_pos = [1, -0.5, 0.3]  # goal is to reach this target

        # Observation is the current joint angles and velocities
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints * 2,), dtype=np.float32)

        # action is the vector of joint velocities
        self.action_space = spaces.Box(low=-3, high=3, shape=(self.num_joints,), dtype=np.float32)

    def _get_joint_states(self):
        state = []
        for j in range(self.num_joints):
            jointPos, jointVel, _, _ = p.getJointState(bodyUniqueId=self.robot, jointIndex=j)
            state += [jointPos, jointVel]
        return np.array(state)

    def reset(self,seed=None, options = None):
        self.t = 0

        for j in range(self.num_joints):
            p.resetJointState(bodyUniqueId=self.robot, jointIndex=j, targetValue=0, targetVelocity=0)

        for _ in range(10):
            p.stepSimulation()  # step a couple of times to make there is no weirdness after resetting

        obs = self._get_joint_states()
        return obs, _

    def step(self, action):
        penalty = 0
        # Execute one time step within the environment
        for j in range(self.num_joints):
            p.setJointMotorControl2(bodyIndex=self.robot, jointIndex=j, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action[j])

        p.stepSimulation()  
        self.t += 1

        obs = self._get_joint_states()
        
        # panalty for high velocity
        for i in range(self.num_joints):
            penalty += np.linalg.norm(np.array(p.getLinkState(self.robot,i)[-2]))*0.1
        
        
        end_effector_pos = p.getLinkState(bodyUniqueId=self.robot, linkIndex=8)[0]
        reward = -sum(abs(pi - ti) for pi, ti in zip(end_effector_pos, self.target_pos))
        
        if np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_pos)) < 0.05:
            reward += 100
            done = True
        else:
          done = (self.t >= 2500)
            
      #  print(reward,penalty)
        reward -=penalty
        return obs, reward, done, done, {}

    def render(self):
        cam_target_pos = [.95, -0.2, 0.2]
        cam_distance = 2.05
        cam_yaw, cam_pitch, cam_roll = -50, -40, 0
        cam_width, cam_height = 480, 360
        # render image
        cam_up, cam_up_axis_idx, cam_near_plane, cam_far_plane, cam_fov = [0, 0, 1], 2, 0.01, 100, 60
        # print(p.getLinkState(bodyUniqueId=robot_id, linkIndex=2)[0])
        cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch,
                                                              cam_roll, cam_up_axis_idx)
        cam_projection_matrix = p.computeProjectionMatrixFOV(cam_fov, cam_width * 1. / cam_height, cam_near_plane,
                                                             cam_far_plane)
        image = p.getCameraImage(cam_width, cam_height, cam_view_matrix, cam_projection_matrix)[2][:, :, :3]
        image = np.ascontiguousarray(image)
        return image


