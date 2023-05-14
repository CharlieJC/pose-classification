import numpy as np
import cv2

def normalize(x,x_min,x_max,interval=(0,1)):
    return interval[0] + (x-x_min)*(interval[1]-interval[0])/(x_max-x_min)
# normalize(50,0,100,(0,256))
#since skeleton function doesnt like 0 values we add a 1 pixel buffer around
def normalize_kpts(kpts, steps, interval=(1,255)):
    num_kpts = len(kpts) // steps
    #maybe hardcode indexes for this eventually for performance
    max1 = 0;
    max2 = 0;
    min1 = 1000; # must be larger than image resolution
    min2 = 1000; 
    for n in range(num_kpts):
        if kpts[n*steps] > max1:
            max1 = kpts[n*steps]
        if kpts[n*steps] < min1:
            min1 = kpts[n*steps]
        if kpts[n*steps+1] > max2:
            max2 = kpts[n*steps+1]
        if kpts[n*steps+1] < min2:
            min2 = kpts[n*steps+1]
                   
    for n in range(num_kpts):
        kpts[n*steps] = normalize(kpts[n*steps],min1,max1,interval)
        kpts[n*steps+1] = normalize(kpts[n*steps+1],min2,max2,interval)

#adapted keypoint function for drawing a single pose skeleton per image
#scale using min-max feature scaling, remove head
def plot_pose_skeleton(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])
    
    skeleton = [[11, 9], [9, 7], [12, 10], [10, 8], [7, 8], [1, 7],
                 [2, 8], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6]] 

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0]]
    
    num_kpts = len(kpts) // steps

    for sk_id, sk in enumerate(skeleton):
        
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
       
        if pos1[0]%960 == 0 or pos1[1]%960==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 960 == 0 or pos2[1] % 960 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
       
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=7)


