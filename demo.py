import data_utils as utils
from imutils.video import FileVideoStream, VideoStream
from imutils.video import FPS
from mtcnn import MTCNN
import time
import cv2
import model
import tensorflow as tf

if __name__ == '__main__':
    arguments = utils.get_arguments()
    image_path = None
    video_path = None
    session = tf.compat.v1.Session()

    # load model multitask learning
    multitask_model = model.Model(session=session, trainable=False, prediction=True)

    # load model detect faces
    detect_model = MTCNN()

    if arguments.image_path is not None:
        image_path = arguments.image_path
    else:
        video_path = arguments.video_path

    if image_path is not None:  # case 1: detect image
        img = cv2.imread(image_path)

        # detect faces
        result = detect_model.detect_faces(img)

        # cropped face
        cropped_face, boxes = utils.crop_face(img, result)

        # predict
        images = (cropped_face - 128.0) / 255.0
        predicted_result = multitask_model.predict(images)

        # draw label and boxes
        output = utils.draw_labels_and_boxes(img, boxes, predicted_result)

        # save image
        cv2.imwrite('test1.png', output)

        # show image

        cv2.imshow('Images', output)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)
    elif video_path is not None:
        # start to the file video stream thread and allow the buffer to
        # start to fill
        print('[INFO] starting video file thread....')
        fvs = FileVideoStream(video_path).start()
        time.sleep(1.0)

        # start the FPS timer
        fps = FPS().start()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

        # loop over frames from the video file stream
        while fvs.more():
            frame = fvs.read()
            if frame is None:
                break
            height, width = frame.shape[:2]

            # detect faces
            result = detect_model.detect_faces(frame)

            # cropped face
            cropped_face, boxes = utils.crop_face(frame, result)

            # predict
            images = (cropped_face - 128.0) / 255.0
            predicted_result = multitask_model.predict(images)

            # draw label and boxes
            frame = utils.draw_labels_and_boxes(frame, boxes, predicted_result)

            # write the flipped frame
            out.write(frame)

            fps.update()

        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # clean up
        cv2.destroyAllWindows()
        fvs.stop()
    else:      # video streaming
        print('[INFO] sampling frames from camera...')
        stream = VideoStream(src=0).start()
        time.sleep(1.0)
        fps = FPS().start()

        # loop over frames from the video stream
        while True:
            frame = stream.read()
            if frame is None:
                break

            height, width = frame.shape[:2]

            # detect faces
            result = detect_model.detect_faces(frame)

            # cropped face
            cropped_face, boxes = utils.crop_face(frame, result)

            # predict
            images = (cropped_face - 128.0) / 255.0
            predicted_result = multitask_model.predict(images)

            # draw label and boxes
            frame = utils.draw_labels_and_boxes(frame, boxes, predicted_result)

            # show video
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)

            fps.update()

        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # clean up
        cv2.destroyAllWindows()
        stream.stop()
