from src import draw_boxes_util, detection_util
import cv2


def run_on_camera(detection_graph, sess):
    capture = cv2.VideoCapture(0)
    while True:

        ret, image = capture.read()
        image_height, image_width = image.shape[:2]
        boxes, scores, classes = detection_util.detect(image, detection_graph, sess)
        image = draw_boxes_util.draw_boxes(image, image_height, image_width, boxes, scores, classes)
        cv2.imshow('Detection', image)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detection_graph, sess = detection_util.load_inference_graph()
    print('detection_graph_loaded')
    run_on_camera(detection_graph, sess)
