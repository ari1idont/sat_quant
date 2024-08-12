import numpy as np
from qiskit import *
from scipy.special import jv
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit.visualization import*
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv



def rate_cal(s,k,n,y,z,dis,ang):
    usnk=np.zeros([s,k])
    gtxmax=np.zeros(s)
    for i1 in range(0,n): 
        beam_deg=np.argmax(y[i1])+1
        gtxmax[i1]=70*np.pi/np.deg2rad(beam_deg)
        usnk[i1]=(2.071/np.sin(beam_deg))*(np.sin(np.deg2rad(ang[i1])))
    gsnktx=np.zeros([s,k])
    for i2 in range(0,s):
        for j1 in range(0,k):
            temp1=jv(1,usnk[i2,j1])/(2*usnk[i2,j1])
            temp2=36*jv(3,usnk[i2,j1])/(pow(usnk[i2,j1],3))
            temp3=temp1+temp2
            gsnktx[i2,j1]=gtxmax[i2]*pow(temp3,2)
    lsk=32.45+20*np.log10(20e9)+20*np.log10(dis)
    #lsk=20*np.log10(lsk)
    hsnk=np.zeros([s,k])
    #gsnktx_db=20*np.log10(gsnktx)
    for i3 in range(0,s):
        for j2 in range(0,k):
            hsnk[i3,j2]=35*gsnktx[i3,j2]/lsk[i3,j2]
    gamma=np.zeros([s,k])
    temp4 = hsnk*43
    sigma=-126.43
    temp5 = hsnk*43+sigma
    for i4 in range(0,s):
        for j3 in range(0,k):
            temp6=np.concatenate((temp5[:i4,j3],temp5[i4+1:,j3]))
            gamma[i4,j3]=temp4[i4,j3]/sum(temp6)
    rsk=4e8*np.log(1+gamma)
    rk=np.zeros(k)
    for i5 in range(0,k):
        temp7 = 0
        for j4 in range(0,s):
            temp7 = temp7 + z[j4,i5]*rsk[j4,i5]
        rk[i5]=temp7
    return rk



#def vqa_ansatz(qc,k,theta,):


def vqa_circuit(k,theta):
    qc=QuantumCircuit(2*k,2*k)

    qc.h(range(2*k))
    for i in range (0,(2*k)-1):
        qc.ryy(theta[i],i,i+1)
        qc.rzx(theta[i+1],i,i+1)
    qc.ryy(theta[len(theta)-2],0,(2*k)-1)
    qc.rzx(theta[len(theta)-1],0,(2*k)-1)
    qc.measure(range(2*k),range(2*k))
    return qc


def encode_array_to_bitstring(array):
    bitstring = ""
    rows, cols = array.shape
    for col in range(cols):
        if array[0, col] == 1:
            bitstring += "00"
        elif array[1, col] == 1:
            bitstring += "01"
        elif array[2, col] == 1:
            bitstring += "10"
        elif array[3, col] == 1:
            bitstring += "11"
    return bitstring

def decode_bitstring_to_array(bitstring):
    k = len(bitstring) // 2
    array = np.zeros((4, k), dtype=int)
    for i in range(k):
        bits = bitstring[2 * i: 2 * i + 2]
        if bits == "00":
            array[0, i] = 1
        elif bits == "01":
            array[1, i] = 1
        elif bits == "10":
            array[2, i] = 1
        elif bits == "11":
            array[3, i] = 1
    return array




def obj(x,y,z,dk,dis,ang):
    z1=decode_bitstring_to_array(x)
    #replace with ratecal
    rk=rate_cal(4,10,4,y,z1,dis,ang)
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

        



def energy(counts,y,z,dk,dis,ang):
    energy = 0
    total_counts= 0
    for meas,meas_count in counts.items():
        obj_for_meas=obj(meas,y,z,dk,dis,ang)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    #print(energy/total_counts)
    return energy/total_counts
    

def black_box_objective(y,z,dk,dis,ang):
    print('apple')
    def f(theta):
        #print('apple')
        #qc=QuantumCircuit(2*15,2*15)

        qc=vqa_circuit(10,theta)
        #qc.draw()
        simulator=AerSimulator()
        qc=transpile(qc,simulator)
        result=simulator.run(qc,shots=1000).result()
        counts =result.get_counts(qc)
        return energy(counts,y,z,dk,dis,ang)
    return f


# def lets_see()

# x=jv(3,207) function for bessel function
# print (x)

def generate_user_space(s,k):
    # Define the size of the space
    space_size = 200  # 200 km x 200 km

    # Define the number of users
    num_users = k

    # Define the mean and standard deviation for the normal distribution
    mean = space_size / 2
    std_dev = space_size / 10  # You can adjust this to control the spread of users

    # Generate random coordinates for the users
    x_coords = np.random.normal(mean, std_dev, num_users)
    y_coords = np.random.normal(mean, std_dev, num_users)

    # Ensure that all coordinates are within the bounds of the space
    x_coords = np.clip(x_coords, 0, space_size)
    y_coords = np.clip(y_coords, 0, space_size)

    # Print the coordinates
    user_coordinates = list(zip(x_coords, y_coords))
    for idx, (x, y) in enumerate(user_coordinates):
        print(f"User {idx + 1}: ({x:.2f}, {y:.2f})")

    # Define the height of the satellites
    satellite_height = 600  # 600 km

    # Define the positions of the satellites along the diagonal
    satellite_positions = [(i * space_size / 3, i * space_size / 3) for i in range(4)]
    print("\nSatellite Positions (on the ground):")
    for idx, (x, y) in enumerate(satellite_positions):
        print(f"Satellite {idx + 1}: ({x:.2f}, {y:.2f})")

    # Calculate the distance and angle matrices
    distances = np.zeros((4, k))
    angles = np.zeros((4, k))

    for s_idx, (sx, sy) in enumerate(satellite_positions):
        for u_idx, (ux, uy) in enumerate(user_coordinates):
            # Calculate the ground distance between satellite and user
            ground_distance = np.sqrt((sx - ux) ** 2 + (sy - uy) ** 2)
            # Calculate the 3D distance
            distance = np.sqrt(ground_distance ** 2 + satellite_height ** 2)
            distances[s_idx, u_idx] = distance
            
            # Calculate the angle between the nadir and the user
            angle = np.arctan2(ground_distance, satellite_height)  # in radians
            angle = np.degrees(angle)  # convert to degrees
            angles[s_idx, u_idx] = angle

    # Print the distance and angle matrices
    print("\nDistance Matrix (km):")
    print(distances)

    print("\nAngle Matrix (degrees):")
    print(angles)

    print(np.shape(distances))
    print(np.shape(angles))

    return user_coordinates, satellite_positions, distances, angles


class SatelliteBeamEnv(gym.Env):
    def __init__(self, N, S, K, Z, d,dis,ang):
        super(SatelliteBeamEnv, self).__init__()
        
        # Number of beams (N), number of satellites (S), number of users (K)
        self.N = N
        self.S = S
        self.K = K
        
        # Satellite-User association matrix (Z), demand vector (d)
        self.Z = Z
        self.d = d
        self.dis=dis
        self.ang=ang
        
        # Flattened action space: Binary vector of length S*N
        self.action_space = spaces.MultiBinary(S * N)
        
        # Observation space: Flattened binary vector of length S*N
        self.observation_space = spaces.MultiBinary(S * N)
        
        # Track the best Y matrix and its reward
        self.best_Y = None
        self.best_reward = float('-inf')
        
    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility (optional)
        if seed is not None:
            np.random.seed(seed)
        
        # Reset Y to a random valid configuration
        self.Y = np.zeros((self.S, self.N), dtype=int)
        for i in range(self.S):
            self.Y[i, np.random.randint(0, self.N)] = 1
        return self.Y.flatten(),{}
    
    def step(self, action):
        # Reshape the flattened action back to matrix form
        self.Y = action.reshape((self.S, self.N))
        
        # Ensure each row has exactly one 1 (constraint enforcement)
        for i in range(self.S):
            if np.sum(self.Y[i]) != 1:
                self.Y[i] = np.zeros(self.N, dtype=int)
                self.Y[i, np.random.randint(0, self.N)] = 1
        
        # Calculate the rate vector r based on Y and Z
        #ratecal
        r=rate_cal(4,10,4,self.Y,self.Z,self.dis,self.ang)
        #r = self.calculate_rate(self.Y, self.Z)
        
        # Calculate the reward as the negative of the difference between the sum of r and sum of d
        reward = -np.sum(abs(r - self.d))
        
        # Update the best Y if this reward is better
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_Y = self.Y.copy()
        
        # In this environment, we'll not define a specific termination condition, so `done` is always False
        done = False
        
        return self.Y.flatten(), reward, done, False, {}

    # def calculate_rate(self, Y, Z):
    #     # Placeholder for rate calculation; in practice, this would involve complex equations
    #     # depending on Y and Z. Here, we'll use a simplified model:
    #     r = np.dot(Y.T, np.dot(Z, np.random.rand(self.K)))  # Simplified rate calculation
    #     return r
    
    def render(self, mode='human'):
        print(f"Current Y matrix:\n{self.Y}")








S=4 #number of satelllites
K=10 #number of users
N=4 #number of beams
beam_deg=[1,2,3,4] #number of beams in degree (3dB beam width)

# y=np.zeros([S,N]) #matrix for s using beam n
# z=np.zeros([S,K]) #matrix for s connecting to k
y = np.random.randint(0, 2, size=(S, N))
z = np.random.randint(0, 2, size=(S, K))


user_pos, sat_pos, distances, angles = generate_user_space(S,K)

# angles_rad=np.deg2rad(angles)
# gtxmax=70*np.pi/np.deg2rad(3)
# usnk=(2.071/np.sin(1))*(np.sin(angles_rad))
# gsnktx=np.zeros([S,K])
# for i in range(0,S):   
#     for j in range(0,K):
#         gsnktx[i,j]=gtxmax*pow(((jv(1,usnk[i,j])/(2*usnk[i,j]))+((36*jv(3,usnk[i,j]))/pow(usnk[i,j],3))),2)
# # print(np.shape(gsnktx))
# # print(gsnktx)

# gsnktx_db=20*np.log10(gsnktx)
# lsk=32.45+20*np.log(20e9)+20*np.log(distances)
# hsnk=np.zeros([S,K])

# for i in range(0,S):
#     for j in range(0,K):
#         hsnk[i,j]=35*gsnktx[i,j]/lsk[i,j]

# gamma = np.zeros([S,K])

# for i in range(0,S):
#     net_sum=sum(sum(hsnk*43))
#     row_sum=sum((hsnk[i]*43)-126)
#     for j in range(0,K):
        
#         gamma[i,j]= hsnk[i,j]/(net_sum-row_sum)


# rsk= 4e8*np.log(1+gamma)
theta =  np.random.uniform(0, 0.05, size=(4*K))
theta=theta*np.pi
dk = 10e6 * np.ones(10) 

#optimization part 
obj1=black_box_objective(y,z,dk,distances,angles)
res_sample = minimize (obj1,theta,method = 'COBYLA', options={'maxiter':1500, 'disp':True})
optimal_theta = res_sample['x']
qc1=vqa_circuit(10,optimal_theta)
simulator=AerSimulator()
qc1=transpile(qc1,simulator)
result=simulator.run(qc1,shots=1500).result()
counts =result.get_counts(qc1)

max_bitstring = max(counts, key=lambda x: counts[x])
# result=obj(max_bitstring,y,z,dk,distances,angles)


print('result: ' + str(result))
z_new=decode_bitstring_to_array(max_bitstring)
print(max_bitstring)
#print(counts)
print(len(counts))
print(optimal_theta)

z_prime=decode_bitstring_to_array(max_bitstring)

env=SatelliteBeamEnv(N,S,K,z_prime,dk,distances,angles)

env=DummyVecEnv([lambda:env])



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

# Print the final and most optimal Y matrix after training
print("Most optimal Y matrix found during training:")
print(env.get_attr('best_Y')[0])
print(f"Best reward: {env.get_attr('best_reward')[0]}")

y_prime=env.get_attr('best_Y')[0]

result=obj(max_bitstring,y_prime,z,dk,distances,angles)

print('result: ' + str(result))
