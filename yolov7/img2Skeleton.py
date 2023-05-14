import sys
sys.path.append('/workspace/yolov7')
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.classification import plot_pose_skeleton,normalize_kpts

def load_model():
   #Load Yolov7-Pose model and COCO trained weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)

        
    return model


def pose_images(filename, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Load input file, resize and transform into PyTorch tensor for inference.
    image = cv2.imread(filename)
    if image is None:
        return
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    #Regressively predict joint keypoints for all people in image
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():    
        output, _ = model(image)

    # Clear memory
    del image, image_, device, _
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    #.... research function more before comment
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
 
    #.... research function more before comment
    with torch.no_grad():
        output = output_to_keypoint(output)
    
    
    #Convert predicted keypoints into
    skeleton_images = []
   #create 256x256 skeletons, normalized
    for idx in range(output.shape[0]):
        temp_kpts = output[idx,22:].T
        skel_img = np.zeros((256,256,3), dtype = np.uint8);
        normalize_kpts(temp_kpts, 3)
        plot_pose_skeleton(skel_img, temp_kpts , 3)
        skeleton_images.append(skel_img)

    del output
    return skeleton_images

# +
# img_file = "/workspace/data/yoga82/download/Cobra_Pose_or_Bhujangasana_/2_62.jpg"

#         %matplotlib inline
# #     plt.figure(figsize=(8,8))
#         plt.axis('off')
#         plt.imshow(im)
#         plt.show()
import os
#directory_ = "/workspace/data/yoga82/download/Cobra_Pose_or_Bhujangasana_"

image_extension = ".jpg"
model = load_model()
def img_to_pose(img_file, model):
    images = pose_images(img_file, model)
    for idx, im in enumerate(images):
        new_file = img_file[:len(img_file)-4] + "_" + str(idx) + "_" + "skel.jpg"
        cv2.imwrite(new_file,im)
import gc        
def img_to_pose_dir(directory, model):
    out_dir = directory + "/skeletons/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

   #FIND OUT WHY 2_151 wont work
   #maybe make it only print complete skeletons, no missing limbs, or few missing limbs
    for file in os.listdir(directory):
        if file.lower().endswith(image_extension):
            if file.lower().endswith("skel.jpg"):
                continue
            print(directory + "/" + file)
            images = pose_images(directory + "/" + file, model)
            if images is None:
                continue
            for idx, im in enumerate(images):
                skel_file = out_dir + file[:len(file)-len(image_extension)] + "_" + str(idx) + "_" + "skel.jpg"
                print(skel_file)
                success = cv2.imwrite(skel_file,im)
                
                if not success:
                    print("Skeleton Generation Fail:", skel_file)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()    



# +
def main():
    poses = ["Cobra_Pose_or_Bhujangasana_","Cat_Cow_Pose_or_Marjaryasana_", "Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_", "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_"]
    directory = "/workspace/data/yoga82/download/"
    directories = [(directory + pose) for pose in poses]
    for directory in directories:
        print("Creating skeletons for: " + directory)
        img_to_pose_dir(directory, model)

if __name__ == "__main__":
    main()    
