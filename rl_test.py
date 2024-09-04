import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from scipy.special import jv

def rate_cal(s, k, n, y, z, dis, ang):
    # Vectorized computation of beam_deg and gtxmax
    beam_deg = np.argmax(y[:n], axis=1) + 1
    gtxmax = np.power(70 * np.pi / np.deg2rad(beam_deg), 2)

    # Vectorized computation of usnk
    usnk = (2.071 / np.sin(np.deg2rad(beam_deg))[:, None]) * np.sin(np.deg2rad(ang[:n]))

    # Vectorized computation of gsnktx
    temp1 = jv(1, usnk) / (2 * usnk)
    temp2 = 36 * jv(3, usnk) / np.power(usnk, 3)
    temp3 = temp1 + temp2
    gsnktx = gtxmax[:, None] * np.power(temp3, 2)

    # Vectorized computation of lsk
    lsk = 32.45 + 20 * np.log10(20e9) + 20 * np.log10(dis * 1000)
    lsk = np.power(10, lsk / 20)

    # Vectorized computation of hsnk
    hsnk = np.power(10, 35 / 20) * gsnktx / lsk

    # Vectorized computation of gamma
    temp4 = hsnk * np.power(10, 43 / 20)
    sigma = np.power(10, -126 / 20)

    # Compute the sum over the axis for each `i`, keeping the correct shape
    temp6_sum = np.sum(temp4, axis=0)  # shape (k,)
    gamma = temp4 / (temp6_sum - temp4 + sigma)  # Broadcasting temp6_sum to subtract each row individually

    # Vectorized computation of rsk and rk
    rsk = 4e8 * np.log2(1 + gamma)
    rk = np.sum(z * rsk, axis=0)

    return rk




def computey(hsnk, zed, s, k, n):
    # Step 1: Calculate temp1 as a 3D array
    temp1 = hsnk * np.expand_dims(zed, axis=-1)  # Shape: (s, k, n)

    # Step 2: Calculate the sums for temp1 and hsnk along the second axis
    sum_temp1 = np.sum(temp1, axis=1)  # Shape: (s, n)
    sum_hsnk = np.sum(hsnk, axis=1)    # Shape: (s, n)

    # Step 3: Calculate nu using broadcasting
    nu = sum_temp1 / (sum_hsnk - sum_temp1)  # Shape: (s, n)

    # Step 4: Initialize y and set the appropriate indices to 1
    y = np.zeros([s, n])
    max_indices = np.argmax(nu, axis=1)  # Shape: (s,)

    # Use advanced indexing to set the appropriate elements to 1
    y[np.arange(s), max_indices] = 1

    return y

    

def rate_cal_3d(s,k,n,dis,ang,y,z,hsnk):
    #dis_expand=np.expand_dims(dis,axis=-1)
    # beams = np.array([1,2,3,4])


    # ang_expand=np.expand_dims(ang,axis=-1)
    # beams = np.array([1,2,3,4])
    # beams_expand=np.expand_dims(beams,axis=(0,1))
    # usnk=2.071 * (np.sin(np.deg2rad(ang_expand)))/(np.sin(np.deg2rad(beams_expand)))
    # gtxmax= np.power(70*np.pi/np.deg2rad(beams),2)
    # gtxmax_expanded = np.expand_dims(gtxmax,axis=(0,1))
    # temp1 = jv(1, usnk) / (2 * usnk)
    # temp2 = 36 * jv(3, usnk) / np.power(usnk, 3)
    # temp3 = temp1 + temp2
    # gsnktx=gtxmax_expanded*temp3
    # lsk = 32.45 + 20 * np.log10(20e9) + 20 * np.log10(dis * 1000)
    # lsk = np.power(10, lsk / 20)
    # lsk_expanded=np.expand_dims(lsk,axis=-1)
    # hsnk=np.power(10,35/20)*gsnktx/lsk_expanded
    y_new=computey(hsnk,z,4,10,4)
    r=rate_cal(s,k,n,y_new,z,dis,ang)
    return r
    


class SatelliteBeamEnv(gym.Env):
    def __init__(self, N, S, K, d, dis, ang, hsnk):
        super(SatelliteBeamEnv, self).__init__()
        
        # Number of beams (N), number of satellites (S), number of users (K)
        self.N = N
        self.S = S
        self.K = K
        
        # Satellite-User association matrix (Z), demand vector (d)
        self.d = d
        self.dis = dis
        self.ang = ang
        self.hsnk = hsnk
        
        # Flattened action space: Binary vector of length (S*N + S*K)
        self.action_space = spaces.MultiBinary(S * N + S * K)
        
        # Observation space: Flattened binary vector of length (S*N + S*K)
        self.observation_space = spaces.MultiBinary(S * N + S * K)
        
        # Track the best Y and Z matrices and their reward
        self.best_Y = None
        self.best_Z = None
        self.best_reward = float('-inf')
        
    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility (optional)
        if seed is not None:
            np.random.seed(seed)
        
        # Reset Y and Z to random valid configurations
        self.Y = np.zeros((self.S, self.N), dtype=int)
        for i in range(self.S):
            self.Y[i, np.random.randint(0, self.N)] = 1
        
        self.Z = np.zeros((self.S, self.K), dtype=int)
        for k in range(self.K):
            self.Z[np.random.randint(0, self.S), k] = 1
        
        return np.concatenate((self.Y.flatten(), self.Z.flatten())), {}
    
    def step(self, action):
        # Split the action into Y and Z parts
        action_Y = action[:self.S * self.N]
        action_Z = action[self.S * self.N:]
        
        # Reshape actions back to matrix form
        self.Y = action_Y.reshape((self.S, self.N))
        self.Z = action_Z.reshape((self.S, self.K))
        
        # Ensure each row in Y has exactly one 1 (constraint enforcement)
        for i in range(self.S):
            if np.sum(self.Y[i]) != 1:
                self.Y[i] = np.zeros(self.N, dtype=int)
                self.Y[i, np.random.randint(0, self.N)] = 1
        
        # Ensure each column in Z has exactly one 1 (constraint enforcement)
        for k in range(self.K):
            if np.sum(self.Z[:, k]) != 1:
                self.Z[:, k] = np.zeros(self.S, dtype=int)
                self.Z[np.random.randint(0, self.S), k] = 1
        
        # Calculate the rate vector r based on Y and Z
        r = rate_cal_3d(4, 10, 4, self.dis, self.ang, self.Y, self.Z, self.hsnk)
        
        # Calculate the reward as the negative of the sum of absolute differences
        reward = -np.sum(abs(r - self.d))
        
        # Update the best Y and Z if this reward is better
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_Y = self.Y.copy()
            self.best_Z = self.Z.copy()
        
        # No specific termination condition, so `done` is always False
        done = False
        
        return np.concatenate((self.Y.flatten(), self.Z.flatten())), reward, done, False, {}

    def render(self, mode='human'):
        print(f"Current Y matrix:\n{self.Y}")
        print(f"Current Z matrix:\n{self.Z}")

def obj(y,z,dk,dis,ang,hsnk):
    #z1=decode_bitstring_to_array(x)
    #replace with ratecal
    rk=rate_cal_3d(4,10,4,dis,ang,y,z,hsnk)
    # rk=np.zeros(10)
    # for i in range(0,10):
    #     temp = 0
    #     for j in range(0,4):
    #         temp = temp + z1[j,i]*rsk[j,i]
    #     rk[i]=temp
    temp=np.zeros(10)
    for i in range(0,10):
        temp[i]=    abs(rk[i]-dk[i])

    return sum(temp)



n=4
s=4
k=10
dist=np.load('distance_dataset.npy')
angl=np.load('angle_dataset.npy')
d=[10e6,20e6,50e6,100e6,200e6,400e6,1000e6,1500e6]
gaps=[]


ang_expand=np.expand_dims(angl[15],axis=-1)
beams = np.array([1,2,3,4])
beams_expand=np.expand_dims(beams,axis=(0,1))
usnk=2.071 * (np.sin(np.deg2rad(ang_expand)))/(np.sin(np.deg2rad(beams_expand)))
gtxmax= np.power(70*np.pi/np.deg2rad(beams),2)
gtxmax_expanded = np.expand_dims(gtxmax,axis=(0,1))
temp1 = jv(1, usnk) / (2 * usnk)
temp2 = 36 * jv(3, usnk) / np.power(usnk, 3)
temp3 = temp1 + temp2
gsnktx=gtxmax_expanded*temp3
lsk = 32.45 + 20 * np.log10(20e9) + 20 * np.log10(dist[15] * 1000)
lsk = np.power(10, lsk / 20)
lsk_expanded=np.expand_dims(lsk,axis=-1)
hsnk=np.power(10,35/20)*gsnktx/lsk_expanded

n=4
s=4
k=10
dist=np.load('distance_dataset.npy')
angl=np.load('angle_dataset.npy')


for j in range(0,8):

    dk=np.ones(10) * d[j]

    env = SatelliteBeamEnv(n, s, k, dk, dist[15], angl[15], hsnk)

    # Wrap the environment in a DummyVecEnv for compatibility with stable-baselines3
    env = DummyVecEnv([lambda: env])

    # Initialize PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("ppo_satellite_beam")

    # Test the trained model
    obs = env.reset()
    for i in range(10):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Reward: {reward}")

    # Print the final and most optimal Y and Z matrices after training
    best_Y = env.get_attr('best_Y')[0]
    best_Z = env.get_attr('best_Z')[0]
    best_reward = env.get_attr('best_reward')[0]

    print("Most optimal Y matrix found during training:")
    print(best_Y)
    print("Most optimal Z matrix found during training:")
    print(best_Z)
    print(f"Best reward: {best_reward}")

    # Assign the optimal Y and Z matrices to variables
    y_prime = best_Y
    z_prime = best_Z

    gap_finale=obj(y_prime,z_prime,dk,dist[15],angl[15],hsnk)
    print(gap_finale)
    gaps.append(gap_finale)

np.save('rl_result_2.npy',gaps)

