import numpy as np
import cv2
import os
from gtts import gTTS

#Loading the labels that are going to be trained in yolo model
object_labels = open("coco.names").read().strip().split("\n")

# Loading yolo-v3 model and trainng it with coco labels
print("Running Yolo-V3 model")
model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Only filtering the layer's name from the output of yolo model
output = model.getLayerNames()
output = [output[j[0] - 1] for j in model.getUnconnectedOutLayers()]

##colors for labeling the object
object_colors = np.random.uniform(0, 255, size=(len(object_labels), 3))
label_font = cv2.FONT_HERSHEY_SIMPLEX

##Starting of the webcam
video = cv2.VideoCapture(0)
cnt_frame = 0
first = True
frames_array = []

while True:
    #Incrementing the frame
    cnt_frame += 1
    sqt, frames = video.read()
    frames = cv2.flip(frames, 1)
    frames_array.append(frames)

    if sqt:
        waiting = cv2.waitKey(1)

        #determing the height and width of the frame
        (Height, Width) = frames.shape[:2]

        #Creating blob around object for object detection and passing to the yolo model to detection
        bboxes = cv2.dnn.blobFromImage(frames, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        model.setInput(bboxes)
        final_layers = model.forward(output)

        #intializing the variables for accuracies and distance measure
        bboxes_1 = []
        Accuraries = []
        label_ids = []
        center_points = []
        distances=[]

        #used for loop to calculate each label with accuracy and distance
        for out in final_layers:
            for object_detection in out:
                accu_score = object_detection[5:]
                label_id = np.argmax(accu_score)
                accuracy = accu_score[label_id]

                # this if condition is used to state that only label with accuracy more than 50% will be displayed.
                if accuracy > 0.5:
                    bbox = object_detection[0:4] * np.array([Width,Height,Width,Height])
                    (midX, midY, w, h) = bbox.astype("int")
                    x_1 = int(midX - (w / 2))
                    y_1 = int(midY - (h / 2))

                    #updating the blobs as the moving object, similar for the label and accuracy.
                    bboxes_1.append([x_1, y_1, int(w), int(h)])
                    Accuraries.append(float(accuracy))
                    label_ids.append(label_id)
                    center_points.append((midX, midY))

                    # calculate the distance with respect to blob height and width
                    distance = ((2 * 3.14 * 180) / (int(w) + int(h) * 360) * 1000 + 3)
                    distances.append(float(distance))

        final_label_id = cv2.dnn.NMSBoxes(bboxes_1, Accuraries, 0.5, 0.3)

        #using this for loop for printing each object and its label with accuracy.
        for i in range(len(bboxes_1)):
            if i in final_label_id:
                label_1, label_2, label_3, label_4 = bboxes_1[i]
                labels = str(object_labels[label_ids[i]])
                accuracy = Accuraries[i]
                distance = distances[i]
                colors = object_colors[label_ids[i]]
                cv2.rectangle(frames, (label_1, label_2), (label_1 + label_4, label_2 + label_3), colors, 2)
                cv2.putText(frames, labels + " " + str(round(accuracy, 2)), (label_1, label_2 + 30), label_font, 3, colors, 3)
                cv2.imshow("Image", frames)

        #using this array to store the parameters to convert them into speech
        speech = []
        if len(final_label_id)>0:
            for i in final_label_id.flatten():
                cen_x, cen_y = center_points[i][0], center_points[i][1]
                distance = distances[i]

                #using if condition to determine where is the object in the lane. This can be done with the help of width and height of the frame.
                if cen_x < Width/3:
                    width_pos = "Left"
                elif cen_x <= (Width/3*2):
                    width_pos = "center "
                else:
                    width_pos = "right "

                if cen_y <= Height/3:
                    height_pos = "top "
                elif cen_y <= (Height/3*2):
                    height_pos = "mid "
                else:
                    height_pos = "bottom "

                #finally appending the results.
                speech.append(height_pos + width_pos + object_labels[label_ids[i]] + " is " + str(round(distance, 2)) + " inches away ")

            print(speech)

            #code for the google translate text to speech.
            if speech:
                description = ', '.join(speech)
                voice = gTTS(description, lang='en')
                voice.save("voice_feedback.mp3")
                os.system("mpg321 voice_feedback.mp3")

video.release()
cv2.destroyAllWindows()





