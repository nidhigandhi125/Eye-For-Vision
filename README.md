# Eye-For-Vision

Thousand of people across the globe suffer from either complete or partial blindness which makes it difficult for them to walk around without any type of assistance. Numerous systems have been already developed to aid visually impaired persons to navigate through but are quite
costly and hefty for a common person to buy. The project Eye for Vision aims to fill in the gap and provide common people with solutions that are accurate and affordable. The project proposes an AI-based method wherein the objects will be detected through a high-resolution camera and the object’s label will be passed onto the user through auditory feedback. The proposed method will detect the objects along with the lane where the object is located in order to assist the user to navigate freely, safely, and confidently.

# Dataset

COCO stands for Common Objects in Context. COCO dataset is large-scale object recognition, segmentation, and captioning dataset. It contains around 1,22,000 images divided into 80 categories. The resolution of the images is around 640x480 pixels. It also contains keypoint detection i.e. humans are labeled with key points for example:- elbow, knee, neck, etc. In these 80 categories, each and every element is tried to be included so that visually impaired people can easily know about their
surroundings. 

# Methodology 

In the project, we are building a DNN model for object detection. The process starts by capturing the object through a webcam, which is then converted into a frame. To label the detected objects, COCO names are used for training the model that would categorize the objects. Next, blobs are detected from the video captured in the webcam which will properly capture the object and build a boundary box surrounding the object. This would help categorize the objects effectively from the trained model. After having the label of the object, the model locates the object's exact position so that any blind person can know about the presence of the object and can react accordingly. To find the location of the object, we are detecting the object in the frame which is divided into three sections to have the approximate position of the object. By analyzing the frame’s length and height, the lane is divided into left, right, and center along with positions top, center, and bottom to correctly locate the object for the users. Along with the lane, the distance of the object from the camera is also determined based on the blob’s dimensions. The distance is considered from the camera’s lens to the detected blobs and is displayed in terms of inches.

<img src="https://github.com/nidhigandhi125/Eye-For-Vision/blob/main/proposed%20architecture.png" />

# Results

We can say that our approach can detect the label correctly most of the time. Certain objects are not detected in minimum lights conditions which can lead to false detection. Suppose, if the cell phone’s backside is facing the camera, sometimes the system will detect it as a remote. Hence, even after achieving all the four features, there are some limitations which can be further improved.
