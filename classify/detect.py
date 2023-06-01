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
from model import SiameseNetwork
from PIL import Image
import json

def load_model():
   #Load Yolov7-Pose model and COCO trained weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('/workspace/yolov7/yolov7-w6-pose.pt', map_location=device)
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

def classify_pose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork(3).to(device)
    model.load_state_dict(torch.load('pose_classify.pt'))
    model.eval()
    
    with open('mean_std.json', 'r') as f:
        mean_std = json.load(f)

    mean = mean_std['mean']
    std = mean_std['std']

    img = pose_images('/workspace/data/yoga82/download/Cobra_Pose_or_Bhujangasana_/1_110.jpg', load_model())[0]
#     img = pose_images('/workspace/data/yoga82/download/Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_/1_523.jpg', load_model())[0]
    img = pose_images('/workspace/data/yoga82/download/Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_/2_96.jpg', load_model())[0]

    # Apply the same transformations as during training
    transform_img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)  # provide the mean and std you used in training
                    ])
    
    image = transform_img(img)
    image = image.unsqueeze(0)  # Add batch dimension. image -> [1, 3, H, W]
    
    image = image.to(device)
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    class_names = ['Kneeling', 'Lying', 'Standing']  # replace with your actual class names
    print('Predicted pose: ', class_names[predicted_class.item()])

def main():
    classify_pose()

if __name__ == "__main__":
    main()    
