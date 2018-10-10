import numpy as np
from helpers import *
from keras.layers import *
from keras.models import Model
from moviepy.editor import VideoFileClip
import tensorflow as tf
from tqdm import tqdm
from lane import *
import cv2


OBJ_THRESHOLD = 0.6
NMS_THRESHOLD = 0.5


IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX = 5
CLASS = 80

from flask import Flask, render_template, Response
import threading
from visualizations import *

new_frame = None
flask_thread = True
app = Flask(__name__)

def create_model():

    # the function to implement the organization layer (github.com/allanzelener/YAD2K)
    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)

    # Define input
    x_input = Input([IMAGE_H, IMAGE_W, 3])

    # Layer 1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(x_input)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
        skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23
    x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)


    output = Reshape([GRID_H, GRID_W, BOX, 4 + 1 + CLASS])(x)

    model = Model(x_input, output)

    print('Model created.')

    return model


# Create model and load weights
yolo = create_model()
load_weights(yolo, 'yolo.weights')
yolo.summary()


# All yolo actions from input to output
def make_yolo(original_image):
    input_image = cv2.resize(original_image, (IMAGE_H, IMAGE_W)) / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    yolo_output = np.squeeze(yolo.predict(input_image))
    boxes = filter_boxes(yolo_output, OBJ_THRESHOLD)
    boxes = non_max_suppress(boxes, NMS_THRESHOLD)
    colours = generate_colors(LABELS)
    output_image = draw_boxes(original_image, boxes, LABELS, colours)

    return output_image


def pipeline_yolo(img):
    img_undist, img_lane_augmented, lane_info = lane_process(cv2.resize(img,(1280, 720)))
    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)
    image = make_yolo(output)
    return image

# Objects detection from image
def yolo_image(image_path):

    original_image = cv2.imread(image_path)
    image = make_yolo(original_image)
    cv2.imshow('frame', image)


# Objects detection from video
def yolo_video(video_path, faster_times=1):
    video_output = 'examples/video_output.mp4'
    clip1 = VideoFileClip("examples/video.mp4").subclip(27,30)
    clip = clip1.fl_image(pipeline_yolo)
    clip.write_videofile(video_output, audio=False)


def gen():
    while True:
        ret, jpeg = cv2.imencode('.jpg', new_frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def vehicle_detection_yolo(image, image_lane, lane_info):
    # set the timer
    start = timer()
    # compute frame per second
    fps = 1.0 / (timer() - start)
    # draw visualization on frame
    yolo_result = draw_results(image, image_lane, None, fps, lane_info)
    return yolo_result

def draw_results(img, image_lane, yolo, fps, lane_info):
    img_cp = img.copy()
    img_cp = draw_background_highlight(img_cp, image_lane, 1280)
    draw_lane_status(img_cp,lane_info)
    return img_cp

if __name__ == "__main__":
    x = 0
    flask_t = threading.Thread(target=app.run, args=('0.0.0.0',))
    flask_t.start()
    image = None
    while(x != 4):
        x = int(input("Press 1 for image, 2 for video, 3 for live stream , 4 to exit window \n"))
        if (x == 1):
            image = input("Enter name of image : ")
            yolo_image(image)
        if (x == 2):
            video = input("Enter name of video : ")
            yolo_video(video)
        if (x == 3):
            print("Press q to exit\n")
            cap = cv2.VideoCapture('http://192.168.43.116:8081')
            while(True):
                ret, frame = cap.read()
                try:
                    img_undist, img_lane_augmented, lane_info = lane_process(cv2.resize(frame,(1280, 720)))
                    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)
                    image = make_yolo(output)
                except:
                    image = frame
                new_frame = image
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    break

