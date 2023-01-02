import numpy as np
import math

def calibrate(imgpoints, objpoints):
    
    # number of views
    views = len(imgpoints)
    
    H = []
    V = []
    
    # For each view
    for view in range(views):
        
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
        
        # Perform SVD and construct H^hat
        _, _, v = np.linalg.svd(A, full_matrices=True)
        h_hat = np.reshape(v[-1,:],(3,3))
        H.append(h_hat)
        
        # Take columns of H_hat as h1,h2
        h1 = h_hat[:,0]
        h2 = h_hat[:,1]
        #h3 = h_hat[:,2]
        
        # Make matrix to solve for S
        V12 = np.array([h1[0]*h2[0], (h1[0]*h2[1])+(h1[1]*h2[0]), h1[1]*h2[1], (h1[2]*h2[0])+(h1[0]*h2[2]), (h1[2]*h2[1])+(h1[1]*h2[2]), h1[2]*h2[2]])
        V11 = np.array([h1[0]*h1[0], (h1[0]*h1[1])+(h1[1]*h1[0]), h1[1]*h1[1], (h1[2]*h1[0])+(h1[0]*h1[2]), (h1[2]*h1[1])+(h1[1]*h1[2]), h1[2]*h1[2]])
        V22 = np.array([h2[0]*h2[0], (h2[0]*h2[1])+(h2[1]*h2[0]), h2[1]*h2[1], (h2[2]*h2[0])+(h2[0]*h2[2]), (h2[2]*h2[1])+(h2[1]*h2[2]), h2[2]*h2[2]])
        V.append(V12)
        V.append(V11 - V22)
    
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

### Mean Square Error Algo
def mse(K, R_sequence, T_sequence, imgpoints, objpoints):
    
    # make numpy
    K = np.array(K)
    R_sequence = np.array(R_sequence)
    T_sequence = np.array(T_sequence)
    imgpoints = np.array(imgpoints)
    objpoints = np.array(objpoints)
    
    # Find Error
    total_err = 0
    count = 0
    for view in range(R_sequence.shape[0]):
        
        # Make RT matrix
        r1 = R_sequence[view,:,0]
        r1 = np.reshape(r1, (3,1))
        r2 = R_sequence[view,:,1]
        r2 = np.reshape(r2, (3,1))
        T_star = np.reshape(T_sequence[view], (3,1))
        RT = np.concatenate((r1,r2,T_star),axis=1)
        
        # K*[R*|T*]
        H = np.dot(K,RT)
        
        # For each point in the view
        for i in range(objpoints[view].shape[0]):
            
            # Make Pi*
            P = [objpoints[view,i,0],objpoints[view,i,1],1]
            P = np.reshape(P, (3,1))
            
            # Project points
            Pi = np.dot(H,P)
            xi_hat = Pi[0]/Pi[2]
            yi_hat = Pi[1]/Pi[2]
            
            # Find difference and error
            xi_diff = imgpoints[view,i,0,0]- xi_hat
            yi_diff = imgpoints[view,i,0,1]- yi_hat
            total_err += xi_diff[0]**2 + yi_diff[0]**2
            count += 1
    
    # Mean Error and Pixel Error
    mean_error = total_err / count
    pixel_error = math.sqrt(mean_error)

    # Return mean of Squared error
    return  (mean_error,pixel_error)