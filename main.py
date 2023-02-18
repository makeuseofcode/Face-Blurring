import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def image_preprocess(frame):
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades
                                          + 'haarcascade_frontalface_default.xml')
    resized_image = cv2.resize(frame, (640, 640))
    gray_image = cv2.cvtColor(resized_image,
                              cv2.COLOR_BGR2GRAY)
    face_rects = face_detector.detectMultiScale(
        gray_image, 1.04, 5, minSize=(20, 20))
    return resized_image, face_rects


def face_blur(resized_frame, face_rects):
    for (x, y, w, h) in face_rects:
        # Specifying the center and radius
        # of the blurring circle
        center_x = x + w // 3
        center_y = y + h // 3
        radius = h // 1

        # creating a black image having similar
        # dimensions as the frame
        mask = np.zeros((resized_frame.shape[:3]), np.uint8)
        # draw a white circle in the face region of the frame
        cv2.circle(mask, (center_x, center_y), radius,
                   (255, 255, 255), -1)
        # blurring whole frame
        blurred_image = cv2.medianBlur(resized_frame, 99)
        # reconstructing the frame:
        # - the pixels from the blurred frame if mask > 0
        # - otherwise, take the pixels from the original frame
        resized_frame = np.where(mask > 0, blurred_image,
                                 resized_frame)
    return resized_frame


def main():
    while True:
        success, frame = cap.read()
        resized_input, face_rects = image_preprocess(frame)
        blurred_image = face_blur(resized_input, face_rects)
        # Diplay the blurred image
        cv2.imshow("Blurred image", cv2.resize(blurred_image,
                                               (500, 500)))
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
