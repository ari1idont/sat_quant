import numpy as np
from scipy.special import jv





def milp_hsnk_cal(n,y,z,dis,ang):
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
    return hsnk


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
    for i in range(0,10):
        hsnk_updated=milp_hsnk_cal(n,y,z,dis,ang)
        z_new = cal_z(hsnk_updated,s,k)
        y_new=computey(hsnk,z_new,4,10,4)
    r=rate_cal(s,k,n,y_new,z,dis,ang)
    return r


def cal_z(x,s,k):
    musk=np.zeros([s,k])
    for i in range(0,s):
        for j in range(0,k):
            temp1 = np.sum(x[:,j]-x[i,j])
            musk[i,j]=x[i,j]/temp1

    z1=np.zeros([s,k])
    for i1 in range(0,k):
        temp2=np.argmax(musk[:,i1])
        z1[temp2,i1]=1
    #print(z1)
    return z1


s=4
k=10
n=4

d=[10e6,20e6,50e6,100e6,200e6,400e6,1000e6,1500e6]
res=[]
# y=np.zeros([s,n])
# for i in range(0,s):
#     y[i,0]=1
dist = np.load('distance_dataset.npy')
angl = np.load('angle_dataset.npy')
ang_expand=np.expand_dims(angl[5],axis=-1)
beams = np.array([1,2,3,4])
beams_expand=np.expand_dims(beams,axis=(0,1))
usnk=2.071 * (np.sin(np.deg2rad(ang_expand)))/(np.sin(np.deg2rad(beams_expand)))
gtxmax= np.power(70*np.pi/np.deg2rad(beams),2)
gtxmax_expanded = np.expand_dims(gtxmax,axis=(0,1))
temp1 = jv(1, usnk) / (2 * usnk)
temp2 = 36 * jv(3, usnk) / np.power(usnk, 3)
temp3 = temp1 + temp2
gsnktx=gtxmax_expanded*temp3
lsk = 32.45 + 20 * np.log10(20e9) + 20 * np.log10(dist[5] * 1000)
lsk = np.power(10, lsk / 20)
lsk_expanded=np.expand_dims(lsk,axis=-1)
hsnk=np.power(10,35/20)*gsnktx/lsk_expanded


for i1 in d:
    avg=[]
    y = np.random.randint(0, 2, size=(s, n))
    z = np.random.randint(0, 2, size=(s, k))

    for j1 in range(0,50):

       
        
        
        rate=rate_cal_3d(s,k,n,dist[5],angl[5],y,z,hsnk)
        temp1=np.sum(abs(rate-i1))
        avg.append(temp1)
    res.append(np.mean(avg))
np.save('milp1.npy',res)
#print(res)


