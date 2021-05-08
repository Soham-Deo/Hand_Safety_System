import tensorflow as tf
import numpy as np
from src import label_map_util
import os


FROZEN_GRAPH_FOLDER = 'frozen_graph'    # path to folder where model graph is stored
MODEL_PATH = os.path.join(FROZEN_GRAPH_FOLDER, 'ssd_lite_frozen_inference_graph.pb')    # Path to Model Graph
LABELS_PATH = os.path.join(FROZEN_GRAPH_FOLDER, 'labelmap.pbtxt')   # Path to labelmap file

NUM_CLASSES = 1     # max number of classes

label_map = label_map_util.load_labelmap(LABELS_PATH)   # Loading the labelmap

# Getting labels as categories from labelmap
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)   # get corresponding category index

category_map = None     # will be used to store the category id to label name mapping


# populates category map
def create_category_id_map(categories):
    global category_map
    category_map = {}
    for instance in categories:
        category_map[instance['id']] = instance['name']
    return


# load tensorflow graph
def load_inference_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    create_category_id_map(categories)
    return detection_graph, sess


# performs detection on a single image
def detect(image, detection_graph, sess):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_expanded})

    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

