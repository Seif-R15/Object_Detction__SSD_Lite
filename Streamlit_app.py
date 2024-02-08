from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer
import cv2
import sys
from PIL import Image
import mss
import time
import serial
import streamlit as st
import tempfile

# Create a serial object with the appropriate settings for the Raspberry Pi's UART interf
# ser = serial.Serial('/dev/serial0', 9600, timeout=1)



# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db

# Initialize the Firebase app with your credentials
# cred = credentials.Certificate('C:/Users/Seif/Downloads/iotcar-3117d-firebase-adminsdk-zpziz-ae12a1d5f7.json')
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://iotcar-3117d-default-rtdb.firebaseio.com/'
# })

# # Get a reference to the root node of your database
# root_ref = db.reference()

# # Get a reference to the DataNode node
# data_node_ref = root_ref.child('DataNode')

# # Generate a new, unique key for the child node
# new_node_ref = data_node_ref.push()

# # Store some data in the new child node
# new_data = {'AI': 'some output data'}
# new_node_ref.set(new_data)

# #Print the name of the new child node to the console
# print(new_node_ref.key)

# if len(sys.argv) < 4:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
#     sys.exit(0)
net_type = 'mb2-ssd-lite'
model_path = 'models/mb2-ssd-lite-mp-0_686.pth'
label_path = 'models/voc-model-labels.txt'

# if len(sys.argv) >= 5:
#     cap = cv2.VideoCapture(sys.argv[4])  # capture from file
# else:
cap = cv2.VideoCapture(0)   # capture from camera
cap.set(3, 1920)
cap.set(4, 1080)


class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-large-ssd-lite':
    net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-small-ssd-lite':
    net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

prev_frame_time = 0
timer = Timer()

st.title('Custom Object Detection using Streamlit')

st.sidebar.title('Custom Object Detection')
use_webcam = st.sidebar.button('Use Webcam')
stop_button = st.button('Stop')
frame_place_holder = st.empty()
if use_webcam:
    while cap.isOpened() and not stop_button:
    
        ret, orig_image = cap.read()
    #     if orig_image is None:
    #         continue
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        timer.start()
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        interval = timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,255), 8)
            cv2.putText(image, label,
                        (int(box[0])+20, int(box[1])-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

            if class_names[labels[i]] == 'person':
                class_type = 'p'
                print(class_type)
            elif class_names[labels[i]] == 'chair':
                class_type = 'ch'
                print(class_type)
            elif class_names[labels[i]] == 'car':
                class_type = 'c'
                print(class_type)
            else:
                print('Unwkon')

                    # Display Position
            print('left edge: ',box[0])
            print('right edge: ',box[2] )

            if int(box[0])>=450 and int(box[2])>=610:
                pos = 'L'
                print(pos)
            elif int(box[0]) in range(40, 450) and int(box[2]) in range(550, 1401) :
                pos = 'Ct'
                print(pos)
            elif int(box[0])<=40 and int(box[2])<=550:
                pos = 'R'
                print(pos)

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(image, pos, (int(box[2]) - text_size[0] - 10, int(box[1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                
            
    
#             output_str = f"class: {class_type}, position: {pos}\n"
#             ser.write(output_str.encode('utf-8'))
            
#             while True:
#                 # Read a line of data from the serial port
#                 data = ser.readline().decode().strip()
    
#                 # Check if the received data contains the character 'L'
#                 if 'L' in data:
#                     # Do something when 'L' is received
#                     print("Received 'L'")

             # Calculate the FPS
            new_frame_time = time.time()
            if prev_frame_time == 0:
                prev_frame_time = new_frame_time
                            # Print the values of the previous and current frames
            print("Previous frame time:", prev_frame_time)
            print("Current frame time:", new_frame_time)
            if new_frame_time == prev_frame_time:
                new_frame_time = new_frame_time + 0.00000000001
            else:

                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time


                # Draw the FPS value on the frame
                cv2.putText(orig_image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
               


            width = 640
            height = 480

            # Resize the image
            resized_image = cv2.resize(orig_image, (width, height))

            # Display the resized image




            cv2.imshow('Resized Frame', resized_image)
            frame_place_holder.image(image, cv2.COLOR_BGR2RGB)
            if cv2.waitKey(1) & 0xFF == ord('s') or stop_button:
                cv2.imwrite('output_image.jpg', resized_image)
                print('Output image saved.')



cap.release()
cv2.destroyAllWindows()

    # Black: (0, 0, 0)
    # White: (255, 255, 255)
    # Red: (255, 0, 0)
    # Green: (0, 255, 0)
    # Blue: (0, 0, 255)
    # Yellow: (255, 255, 0)
    # Cyan: (0, 255, 255)
    # Magenta: (255, 0, 255)
