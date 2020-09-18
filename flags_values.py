# Utils
from absl import app, flags, logging
FLAGS = flags.FLAGS

flags.DEFINE_string('weights', './checkpoints/custom-tiny-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', ['./data/images/bateau_1.jpg'], 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_integer('num_objects', 2, 'number of  detected objects')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_string('areas', 'game_area_coords.txt', 'path to game area coordinates .txt file')
flags.DEFINE_string('video', './data/video/test_tangram.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output_video', './detections/output.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('show', True, 'show video output')
flags.DEFINE_string('classes', './data/classes/custom.names', 'path to classes file')
flags.DEFINE_integer('num_classes', 12, 'number of classes in the model')
flags.DEFINE_string('output_file', './detections/', 'path to output file containing object coordinates and classe predictions') # {'object_position':{'x1': coord_1, 'y1':coord_2, 'x2':coord_3,'y2':coord_4}, 'object_prediction':{'class_1':proba_1, 'class_2':proba_2, etc...}}
flags.DEFINE_integer('sort_by', 5, 'if sort_by Flag is true then probabilities will be sorted by descending order. The number of returned classes is specified by the  max_classes flag')
flags.DEFINE_integer('margin', 4, ' number of bounding box object corner points (soft-->  k<4 or hard margin --> k=4) that have to belong to the game area in order to be selected ')
