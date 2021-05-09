from src import draw_boxes_util, detection_util
import cv2
import numpy as np
import time


def run_on_camera(detection_graph, sess):
    capture = cv2.VideoCapture(0)
    prev_frame_time = time.time()
    while True:
        ret, image = capture.read()
        image = np.array(np.fliplr(image))
        image_height, image_width = image.shape[:2]
        boxes, scores, classes = detection_util.detect(image, detection_graph, sess)
        image = draw_boxes_util.draw_boxes(image, image_height, image_width, boxes, scores, classes)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(img=image,
                    text=str('FPS: {}').format(int(fps)),
                    org=(2, 15),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    color=(255, 255, 255),
                    fontScale=.50,
                    thickness=1,
                    lineType=1)
        cv2.imshow('Detection', image)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detection_graph, sess = detection_util.load_inference_graph()
    run_on_camera(detection_graph, sess)
