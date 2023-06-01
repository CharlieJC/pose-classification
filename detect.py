#input the raw image and retrun skeleton image for pose classification
def process_skeleton(image):
    pass

#input skeleton image and return pose classification
def classify_pose(skeleton_img):
    pass

#draw on image based on pose classification
def draw_pose_classif(image, poses):
    pass

#input raw image and return PPE classifications
def process_ppe(image):
    pass

#draw on image based on PPE detections
def draw_ppe_detections(image, ppe_detections):
    pass

#input raw image
def detect(image):
    #handle pose classif
    skeleton_img = process_skeleton(image)
    poses = classify_pose(skeleton_image)

    #handle PPE detections
    ppe_detections = process_ppe_detections(image)

    #draw on image (from video or still image)
    draw_pose_classif()
    draw_ppe_detections()
