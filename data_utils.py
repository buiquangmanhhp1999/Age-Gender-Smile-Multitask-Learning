import numpy as np
import cv2
import argparse

from config import SMILE_FOLDER, AGE_AND_GENDER_FOLDER
import config as cf


def getSmileImage(trainable):
    print('==================================================================')
    print('\nLoading smile image datasets.....')
    X1 = np.load(SMILE_FOLDER + 'train.npy', allow_pickle=True)
    X2 = np.load(SMILE_FOLDER + 'test.npy', allow_pickle=True)
    print('Done! ')

    train_data = []
    test_data = []

    for i in range(X1.shape[0]):
        train_data.append(X1[i])

    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    if trainable:
        print('Number of smile train data: ', str(len(train_data)))
    else:
        print('Number of smile test data: ', str(len(test_data)))

    return train_data, test_data


def getAgeImage(trainable):
    print('==================================================================')
    print('\nLoading age image datasets.....')
    X1 = np.load(AGE_AND_GENDER_FOLDER + 'train_age.npy', allow_pickle=True)
    X2 = np.load(AGE_AND_GENDER_FOLDER + 'test_age.npy', allow_pickle=True)

    train_data = []
    test_data = []

    for i in range(X1.shape[0]):
        train_data.append(X1[i])

    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done!')
    if trainable:
        print('Number of age train data: ', str(len(train_data)))
    else:
        print('Number of age test data: ', str(len(test_data)))

    return train_data, test_data


def getGenderImage(trainable):
    print('==================================================================')
    print('\nLoading gender image datasets.....')
    X1 = np.load(AGE_AND_GENDER_FOLDER + 'train_gender.npy', allow_pickle=True)
    X2 = np.load(AGE_AND_GENDER_FOLDER + 'test_gender.npy', allow_pickle=True)

    train_data = []
    test_data = []

    for i in range(X1.shape[0]):
        train_data.append(X1[i])

    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done!')
    if trainable:
        print('Number of gender train data: ', str(len(train_data)))
    else:
        print('Number of gender test data: ', str(len(test_data)))

    return train_data, test_data


def draw_labels_and_boxes(img, boxes, labels, margin=0):
    for i in range(len(labels)):
        # get the bounding box coordinates
        left, top, right, bottom = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        width = right - left
        height = bottom - top
        img_h, img_w = img.shape[:2]

        x1 = max(int(left - margin * width), 0)
        y1 = max(int(top - margin * height), 0)
        x2 = min(int(right + margin * width), img_w - 1)
        y2 = min(int(bottom + margin * height), img_h - 1)

        # Color red
        color = (0, 0, 255)

        # classify label according to result
        smile_label, age_label, gender_label = labels[i]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = '{} {} {}'.format(gender_label, age_label, smile_label)
        cv2.putText(img, text, (left - 35, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return img


def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='path to the image')
    arg.add_argument('-v', '--video_path', help='path to the video file')
    arg.add_argument('-m', '--margin', help='margin around face', default=0.0)
    return arg.parse_args()


def crop_face(image, result):
    nb_detected_faces = len(result)

    cropped_face = np.empty((nb_detected_faces, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 1))
    boxes = []
    # loop through detected face
    for i in range(nb_detected_faces):
        # coordinates of boxes
        bounding_box = result[i]['box']
        left, top = bounding_box[0], bounding_box[1]
        right, bottom = bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]

        # coordinates of cropped image
        x1_crop = max(int(left), 0)
        y1_crop = max(int(top), 0)
        x2_crop = int(right)
        y2_crop = int(bottom)

        face = image[y1_crop:y2_crop, x1_crop:x2_crop, :]
        face = cv2.resize(face, (cf.IMAGE_SIZE, cf.IMAGE_SIZE), cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.reshape(cf.IMAGE_SIZE, cf.IMAGE_SIZE, 1)

        cropped_face[i, :, :, :] = face
        boxes.append((x1_crop, y1_crop, x2_crop, y2_crop))

    return cropped_face, boxes
