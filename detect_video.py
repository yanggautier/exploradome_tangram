# Utils
import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import flags_values

# Deep Learning
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import operator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Yolo
import core.utils as utils
from core.yolov4 import filter_boxes
from core.output import output_box
from core.output import sort_by
from core.output import filter_game_areas
from core.output import format_game_areas

# Image manipulation
from PIL import Image
import cv2
import numpy as np

# comment out below line to enable tensorflow outputs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# if memory growth is enabled for a PhysicalDevice, 
# the runtime initialization will not allocate all memory on the device.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def detect_video(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = FLAGS.size
    video_path = FLAGS.video
    k = FLAGS.margin  
    max_classes = FLAGS.sort_by
    num_objects= FLAGS.num_objects
    game_area_coords_path = FLAGS.areas

    # game areas in normalized format
    game_areas=[]
    with open(game_area_coords_path) as file:
        lines=file.readlines()[1:]
        for line in lines:  
            number_string=line.strip()
            number_string = number_string.split(',')
            number_list = [float(i) for i in number_string]
            game_areas.append(number_list)

    #### saved file path and name
    box_preds_file = open(FLAGS.output_file + 'boxes_and_predictions.txt', 'w')
    area_box_preds_file = open(FLAGS.output_file + 'area_boxes_and_predictions.txt', 'w')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output_video, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()

        # preprocessing
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # end preprocessing
        
        # record time to compute the FPS value
        start_time = time.time()

        # run inference
        image, class_predicted, bp = utils.get_inference(infer, image_data, frame)
        image = np.asarray(image)
        
        if np.any(bp["bboxes"]) : 
            boxes_and_preds=output_box(bp["all_boxes"], bp["all_preds"], bp["boxes"], bp["bboxes"], class_names)

            # get predictions and metrics
            # sorted the predictions by descending order and keep 'max_classes' 
            # number of them
            if FLAGS.sort_by :
                boxes_and_preds = sort_by(boxes_and_preds, max_classes)

            # write boxes coordinates and classe predictions into a file and 
            # save it 
            box_preds_file.write(f"{boxes_and_preds} \n")

            # format bounding game areas from normalized to image-formatted
            height, width, _ = frame.shape
            formatted_game_areas=format_game_areas(game_areas, height, width)

            # filtering box coordinates according to the game areas  
            if FLAGS.margin : 
                area_boxes_and_preds=filter_game_areas(boxes_and_preds,
                                                        formatted_game_areas,
                                                        k)

            # write box coordinates and classe predictions into a file 
            # according to the game areas and save it 
            area_box_preds_file.write(f"{area_boxes_and_preds} \n")
            
            # get fps
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            print("##############################")
        
        # show image
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if FLAGS.show:
            cv2.imshow("result", result)
        
        # save image into a mp4
        if FLAGS.output:
            out.write(result)

        # wait for key press to close the stream
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(detect_video)
    except SystemExit:
        pass