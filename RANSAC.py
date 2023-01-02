import numpy as np
import yaml
import algorithm_impl as algo
import argparse
import random
import math

### TAKE 2D-2D MAPPING AND CONFIG FILES INPUT AS PASSED ARGUMENT
parser = argparse.ArgumentParser(description = 'This is the RANSAC!')
parser.add_argument('-f1', metavar = 'Enter file-name having extracted feature points', type = str, nargs = '?', default = '3D_to_2D_mapping_Noise1.yaml')
parser.add_argument('-f2', metavar = 'Enter file-name having configuration related to RANSAC', type = str, nargs = '?', default = 'RANSAC.config')
args = parser.parse_args()


def readFile():
    filename = args.f1
    with open(filename) as c:
        data = yaml.load(c, Loader=yaml.FullLoader)
    return data    
    
def getConfig():
    configname = args.f2
    with open(configname, 'r') as c:
        prob = float(c.readline().split()[0])
        kmax = int(c.readline().split()[0])
        nmin = int(c.readline().split()[0])
        nmax = int(c.readline().split()[0])
    return prob, nmin, nmax, kmax

def getMatrixA(imgpoint, objpoint):
    # number of calibration points 
    m = len(imgpoint)
    
    # make A matrix of 2m*9 shape 
    A = np.zeros((2*m,9), np.float32)
    
    # assign value to A
    for i in range(m):
        A[2*i,0] = A[(2*i)+1,3] = objpoint[i,0]
        A[2*i,1] = A[(2*i)+1,4] = objpoint[i,1]
        A[2*i,2] = A[(2*i)+1,5] = 1
        A[2*i,6] = -imgpoint[i,0,0] * objpoint[i,0]
        A[2*i,7] = -imgpoint[i,0,0] * objpoint[i,1]
        A[2*i,8] = -imgpoint[i,0,0] * 1
        A[(2*i)+1,6] = -imgpoint[i,0,1] * objpoint[i,0]
        A[(2*i)+1,7] = -imgpoint[i,0,1] * objpoint[i,1]
        A[(2*i)+1,8] = -imgpoint[i,0,1] * 1

    return A

def getMatrixH(A):
    _, _, v = np.linalg.svd(A, full_matrices=True)
    H_hat = np.reshape(v[-1,:],(3,3))
    return H_hat

def getMatrixV(H):
    V = []
    h1 = H[:,0]
    h2 = H[:,1]
    
    # Make matrix to solve for S
    V12 = np.array([h1[0]*h2[0], (h1[0]*h2[1])+(h1[1]*h2[0]), h1[1]*h2[1], (h1[2]*h2[0])+(h1[0]*h2[2]), (h1[2]*h2[1])+(h1[1]*h2[2]), h1[2]*h2[2]])
    V11 = np.array([h1[0]*h1[0], (h1[0]*h1[1])+(h1[1]*h1[0]), h1[1]*h1[1], (h1[2]*h1[0])+(h1[0]*h1[2]), (h1[2]*h1[1])+(h1[1]*h1[2]), h1[2]*h1[2]])
    V22 = np.array([h2[0]*h2[0], (h2[0]*h2[1])+(h2[1]*h2[0]), h2[1]*h2[1], (h2[2]*h2[0])+(h2[0]*h2[2]), (h2[2]*h2[1])+(h2[1]*h2[2]), h2[2]*h2[2]])

    V.append(V12)
    V.append(V11 - V22)
    return V
    

def getDistance(imgpoint, objpoint, H):
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    d = []
    
    for i in range(len(imgpoint)):
        xi = imgpoint[i,0,0]
        yi = imgpoint[i,0,1]
        
        pi = objpoint_noise[i]
        pi = np.delete(pi,-1)
        pi = np.append(pi, 1)        
        
        exi = (h1.T.dot(pi)) / (h3.T.dot(pi))
        eyi = (h2.T.dot(pi)) / (h3.T.dot(pi))
        di = np.sqrt(((xi - exi) ** 2 + (yi - eyi) ** 2))
        d.append(di)
    return d



def ransac(imgpoint, objpoint, prob, nmin, nmax, kmax):

    num_of_inliers = 0
    count = 0
    w = 0.5
    k = kmax    
    bestH = None
    
    A = getMatrixA(imgpoint, objpoint)
    H = getMatrixH(A)
    distance_All = getDistance(imgpoint, objpoint, H)
    medianDistance = np.median(distance_All)
    threshold = 1.5 * medianDistance
    n = random.randint(nmin, nmax)
    while(count < k and count < kmax):
        
        index = np.random.choice(len(objpoint), n)
        randomObjectPoint, randomImagePoint = objpoint[index], imgpoint[index]
        A = getMatrixA(randomImagePoint, randomObjectPoint)
        H = getMatrixH(A)
        d = getDistance(imgpoint, objpoint, H)
        inlier = []
        for i, d in enumerate(d):
            if d < threshold:
                inlier.append(i)
        if len(inlier) >= num_of_inliers:
            num_of_inliers = len(inlier)
            inlierObjectPoint, inlierImagePoint = objpoint[inlier], imgpoint[inlier]
            A = getMatrixA(randomImagePoint, randomObjectPoint)
            bestH = getMatrixH(A)
        if not (w == 0 ):
            w = float(len(inlier))/float(len(imgpoint))
            k = float(math.log(1 - prob)) / np.absolute(math.log(1 - (w ** n)))
        count += 1
        
    return bestH, num_of_inliers

def calibrate_with_ransac(imgpoints, objpoints, prob, nmin, nmax, kmax):
    views = len(imgpoints)
    H = []
    V = []
    
    imgpoint_noise = imgpoints[0]
    objpoint_noise = objpoints[0]
    
    H1_best, num_of_inliers =  ransac(imgpoint_noise, objpoint_noise, prob, nmin, nmax, kmax)
    H.append(H1_best)
    
    V.extend(getMatrixV(H1_best))
    
    for view in range(1, views):
        # number of calibration points in this view
        m = len(imgpoints[view])
        
        # make A matrix of 2m*9 shape for ith view
        A = np.zeros((2*m,9), np.float32)
        
        # assign value to A
        for i in range(m):
            A[2*i,0] = A[(2*i)+1,3] = objpoints[view,i,0]
            A[2*i,1] = A[(2*i)+1,4] = objpoints[view,i,1]
            A[2*i,2] = A[(2*i)+1,5] = 1
            A[2*i,6] = -imgpoints[view,i,0,0] * objpoints[view,i,0]
            A[2*i,7] = -imgpoints[view,i,0,0] * objpoints[view,i,1]
            A[2*i,8] = -imgpoints[view,i,0,0] * 1
            A[(2*i)+1,6] = -imgpoints[view,i,0,1] * objpoints[view,i,0]
            A[(2*i)+1,7] = -imgpoints[view,i,0,1] * objpoints[view,i,1]
            A[(2*i)+1,8] = -imgpoints[view,i,0,1] * 1
     
        # Perform SVD and construct H^
        h_hat = getMatrixH(A) 
        H.append(h_hat)
        
        V.extend(getMatrixV(h_hat))
        
    # Convert to Numpy Array
    H = np.array(H)
    V = np.array(V)
    
    # Find Unknown S values
    _, _, v = np.linalg.svd(V, full_matrices=True)
    s = v[-1,:]
    
    # Find Intrinsic Parameters
    c1 = (s[1]*s[3]) - (s[0]*s[4])
    c2 = (s[0]*s[2]) - (s[1]**2)
    v0 = c1/c2
    lamb = s[5] - ((s[3]**2) + (v0*c1))/s[0]
    alp_u = math.sqrt(lamb/s[0])
    alp_v = math.sqrt((lamb*s[0])/c2)
    skew = -(s[1]*(alp_u**2)*alp_v)/lamb
    u0 = ((skew*v0)/alp_u) - ((s[3]*alp_u**2)/lamb)
    
    # Construct K* from found values
    K = np.zeros((3,3), np.float32)
    K[0,0] = alp_u
    K[1,1] = alp_v
    K[2,2] = 1
    K[0,2] = u0
    K[1,2] = v0
    #K[0,1] = skew
    K_inv = np.linalg.inv(K)
    
    # To store for each view
    R_sequence = []
    T_sequence = []
    
    # Find R* and T* for each view
    for view in range(views):
        
        # Homography for view
        h_hat = H[view,]
        h1 = h_hat[:,0]
        h2 = h_hat[:,1]
        h3 = h_hat[:,2]
        
        # Find Alpha
        mod_alpha = 1/np.linalg.norm(np.dot(K_inv,h1))
        sign = np.dot(K_inv,h3)
        sign = np.sign(sign[2])
        alpha = sign*mod_alpha
        
        # Find R*
        r1 = alpha*(np.dot(K_inv,h1))
        r2 = alpha*(np.dot(K_inv,h2))
        r3 = np.cross(r1,r2)
        r1 = np.reshape(r1, (3,1))
        r2 = np.reshape(r2, (3,1))
        r3 = np.reshape(r3, (3,1))
        R = np.concatenate((r1,r2,r3),axis=1)
        R_sequence.append(R)
        
        # Find T*
        T = alpha*(np.dot(K_inv,h3))
        T_sequence.append(T)
    
    return (K, R_sequence, T_sequence)    
        
        

###############################################################################
data = readFile()
    
imgpoints = np.array(data['imgpoints'], dtype='f')
objpoints = np.array(data['objpoints'], dtype='f')

imgpoint_noise = imgpoints[0]
objpoint_noise = objpoints[0]

prob, nmin, nmax, kmax = getConfig()
A1 = getMatrixA(imgpoint_noise, objpoint_noise)
H1 = getMatrixH(A1)

bestH, num_of_inliers =  ransac(imgpoint_noise, objpoint_noise, prob, nmin, nmax, kmax)
V = getMatrixV(bestH)

K, R_sequence, T_sequence = calibrate_with_ransac(imgpoints, objpoints, prob, nmin, nmax, kmax)

mean_error, pixel_error = algo.mse(K, R_sequence, T_sequence, imgpoints, objpoints)

print(f'Known Parameters:\n----------------')
print(f'(u0,v0): {(K[0,2],K[1,2])}')
print(f'(alphaU,alphaV): {(K[0,0],K[1,1])}')
print(f's: {K[0,1]}')

for i in range(len(R_sequence)):
    print(f'\nImage {i}')
    print(f'T*: {T_sequence[i]}')
    print(f'R*:\n{R_sequence[i]}\n----------------')

print(f'\nMean Square Error: {mean_error}')
print(f'Pixel Error: {pixel_error}')

# transform the object points and image points into readable yaml list
data = {'K*': {'u0': float(K[0,2]),'v0': float(K[1,2]),'alphaU': float(K[0,0]),'alphaV': float(K[1,1]),'skew': float(K[0,1])},
        'R*_all': np.asarray(R_sequence).tolist(),
        'T*_all': np.asarray(T_sequence).tolist(),
        'MSE': float(mean_error),
        'PIXEL_ERR': pixel_error}

# and save it to a file
with open("ransac_output.yaml", "w") as f:
    yaml.dump(data, f)
    print(f'\nCalibration data saved at {f.name}')

