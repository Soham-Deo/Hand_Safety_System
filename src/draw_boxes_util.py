import cv2


def draw_line(image, image_height, image_width):
    """
    Creates Warning and Stop Lines on image and returns the y-coordinate position of those lines
    :return: edited image, y-coordinate of warning and stop lines
    """
    draw_at_ratio = .30

    p1 = (1, int(image_height * draw_at_ratio))
    p2 = (image_width-1, int(image_height * draw_at_ratio))
    warning_line_y = int(image_height * draw_at_ratio)

    cv2.line(img=image,
             pt1=p1,
             pt2=p2,
             color=(255, 0, 0),
             thickness=2,
             lineType=1)

    cv2.putText(img=image,
                text='WARNING Line',
                org=(p1[0], p1[1]-5),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                lineType=1)

    stop_line_p1 = (1, int(image_height * draw_at_ratio/2))
    stop_line_p2 = (image_width - 1, int(image_height * draw_at_ratio/2))
    stop_line_y = int(image_height * draw_at_ratio/2)

    cv2.line(img=image,
             pt1=stop_line_p1,
             pt2=stop_line_p2,
             color=(0, 0, 255),
             thickness=2,
             lineType=1)

    cv2.putText(img=image,
                text='STOP Line',
                org=(stop_line_p1[0], stop_line_p1[1] - 5),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                lineType=1)

    return image, warning_line_y, stop_line_y


def draw_boxes(image, image_height, image_width, boxes, scores, classes):
    """
    Draws bounding boxes on the image
    Checks if hand is crossing the warning line or stopping line
    :return:
    """
    image, warning_line_y, stop_line_y = draw_line(image, image_height, image_width)
    threshold = 0.5
    hand_count = 0
    for i in range(len(scores)):
        hand_count += 1
        if scores[i] > threshold:
            color = (0, 255, 40)

            box = boxes[i]
            top = box[0] * image_height
            left = box[1] * image_width
            bottom = box[2] * image_height
            right = box[3] * image_width

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            cv2.rectangle(img=image,
                          pt1=p1,
                          pt2=p2,
                          color=color,
                          thickness=2,
                          lineType=1)

            cv2.putText(img=image,
                        text=str("HAND - ") + str('{:.2f}').format(scores[i]),
                        org=(int(left), int(top)-5),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=0.5,
                        color=color,
                        lineType=1)

    return image
