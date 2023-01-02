import numpy as np
import yaml
import algorithm_impl as algo
import argparse

### TAKE 2D-2D MAPPING INPUT AS PASSED ARGUMENT
parser = argparse.ArgumentParser(description = 'This is the Calibrator!')
parser.add_argument('-f', metavar = 'Enter file-name having extracted feature points', type = str, nargs = '?', default = 'extracts.yaml')
args = parser.parse_args()

# import and use the extracted data in 1 to calibrate
with open(args.f) as c:
    data = yaml.load(c, Loader=yaml.FullLoader)
print(f'Reading feature points from file: {args.f}\n')
imgpoints = np.array(data['imgpoints'], dtype='f')
objpoints = np.array(data['objpoints'], dtype='f')

K, R_sequence, T_sequence = algo.calibrate(imgpoints,objpoints)

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
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)
    print(f'\nCalibration data saved at {f.name}')