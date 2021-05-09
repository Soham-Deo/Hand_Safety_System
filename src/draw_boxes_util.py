import cv2
from src import check_distance
from src import alert
import yaml


with open('params.yaml') as f:
    config = yaml.safe_load(f)


def draw_line(image, image_height, image_width):
    """
    Creates Warning and Stop Lines on image and returns the y-coordinate position of those lines
    :return: edited image, y-coordinate of warning and stop lines
    """
    draw_at_ratio = float(config['warning_line_ratio'])

    # Warning Line
    p1 = (1, int(image_height * draw_at_ratio))
    p2 = (image_width-1, int(image_height * draw_at_ratio))
    warning_line_y = int(image_height * draw_at_ratio)

    warning_line_color = (int(config['colors']['warning_line']['b']),
                          int(config['colors']['warning_line']['g']),
                          int(config['colors']['warning_line']['r']))

    cv2.line(img=image,
             pt1=p1,
             pt2=p2,
             color=warning_line_color,
             thickness=2,
             lineType=1)

    cv2.putText(img=image,
                text='WARNING Line',
                org=(p1[0], p1[1]-5),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                color=warning_line_color,
                lineType=1)

    stop_line_p1 = (1, int(image_height * draw_at_ratio/2))
    stop_line_p2 = (image_width - 1, int(image_height * draw_at_ratio/2))
    stop_line_y = int(image_height * draw_at_ratio/2)

    # Stop Line
    stop_line_color = (int(config['colors']['stop_line']['b']),
                       int(config['colors']['stop_line']['g']),
                       int(config['colors']['stop_line']['r']))

    cv2.line(img=image,
             pt1=stop_line_p1,
             pt2=stop_line_p2,
             color=stop_line_color,
             thickness=2,
             lineType=1)

    cv2.putText(img=image,
                text='STOP Line',
                org=(stop_line_p1[0], stop_line_p1[1] - 5),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                lineType=1)

    return image, warning_line_y, stop_line_y


def draw_boxes(image, image_height, image_width, boxes, scores, classes):
    """
    Draws bounding boxes on the image
    Checks if hand is crossing the warning line or stopping line
    :return:
    """
    # draw warning and stop line on image and get the y coordinates for both lines
    image, warning_line_y, stop_line_y = draw_line(image, image_height, image_width)
    threshold = float(config['threshold'])

    # check which bbox has score more than threshold
    for i in range(len(scores)):
        if scores[i] > threshold:
            color = (0, 255, 40)

            box = boxes[i]
            top = box[0] * image_height
            left = box[1] * image_width
            bottom = box[2] * image_height
            right = box[3] * image_width

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            alert_warning = check_distance.check(int(top), warning_line_y)

            if alert_warning is True:
                alert.play()
                cv2.rectangle(img=image,
                              pt1=p1,
                              pt2=p2,
                              color=(0, 0, 255),
                              thickness=2,
                              lineType=1)
                cv2.putText(img=image,
                            text="WARNING !!",
                            org=(int(left) + 5, int(top) + int(bottom-top)//2),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=.7,
                            color=(0, 0, 255),
                            lineType=1,
                            thickness=2)

            else:
                cv2.rectangle(img=image,
                              pt1=p1,
                              pt2=p2,
                              color=color,
                              thickness=2,
                              lineType=1)

                cv2.putText(img=image,
                            text=str("HAND - ") + str('{:.2f}').format(scores[i]),
                            org=(int(left), int(bottom)+15),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=0.5,
                            color=color,
                            lineType=1)

    return image
