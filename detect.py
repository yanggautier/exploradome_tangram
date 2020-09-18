# Deep Learning
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

# Utils
from absl import app, flags, logging
import flags_values

# Image manipulation & YOLO
import cv2
import numpy as np
import core.utils as utils
from core.yolov4 import filter_boxes



FLAGS = flags.FLAGS

# if memory growth is enabled for a PhysicalDevice, 
# the runtime initialization will not allocate all memory on the device.

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def detect_img(_argv):
    # Tensorflow Session
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # get application params
    input_size = FLAGS.size
    images = FLAGS.images

    # load model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images):
        
        # preprocessing
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        # end preprocessing

        # run inference
        image, class_predicted, _= utils.get_inference(infer, images_data, original_image)

        # show image and export it
        image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        title = f"{FLAGS.output}detection-{class_predicted}-{str(count)}.png"  
        cv2.imwrite(title, image)

if __name__ == '__main__':
    try:
        app.run(detect_img)
    except SystemExit:
        pass
