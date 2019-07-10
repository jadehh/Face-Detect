import numpy as np
import time

from mtcnn import detect_face, create_mtcnn
import tensorflow as tf

from Face import Face
from scipy import misc
from skimage import transform as trans
import cv2
import math

face_crop_margin = 32
face_crop_size = 160
PRETREINED_MODEL_DIR = './model'

def _setup_mtcnn():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        with sess.as_default():
            return create_mtcnn(sess, PRETREINED_MODEL_DIR)


pnet, rnet, onet = _setup_mtcnn()


def img_to_np(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def alignment(img,bb,landmark,image_size):
  M = None
  # if len(image_size)>0:
  #   image_size = [int(x) for x in image_size.split(',')]
  #   if len(image_size)==1:
  #      image_size = [image_size[0], image_size[0]]

  if landmark is not None:
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

  if M is None:
     ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
     if len(image_size)>0:
        ret = cv2.resize(ret, (image_size[1], image_size[0]))
     return ret
  else:
     warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
     return warped

def calculate_euler(img, landmark):
    model_points = np.array([
        [-165.0, 170.0, -115.0],  # Left eye left corner
        [165.0, 170.0, -115.0],  # Right eye right corne
        [0.0, 0.0, 0.0],  # Nose tip
        [-150.0, -150.0, -125.0],  # Left Mouth corner
        [150.0, -150.0, -125.0]], dtype=np.float32)  # Right mouth corner

    focal_length = img.shape[1]
    center = (img.shape[1] / 2, img.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float32
    )

    dst = landmark.astype(np.int32)
    dst = dst.astype(np.float32)
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, dst, camera_matrix, dist_coeffs)

    rotation3_3 = cv2.Rodrigues(rotation_vector)[0]

    q0 = np.sqrt(1 + rotation3_3[0][0] + rotation3_3[1][1] + rotation3_3[2][2]) / 2
    q1 = (rotation3_3[2][1] - rotation3_3[1][2]) / (4 * q0)
    q2 = (rotation3_3[0][2] - rotation3_3[2][0]) / (4 * q0)
    q3 = (rotation3_3[1][0] - rotation3_3[0][1]) / (4 * q0)

    yaw = math.asin(2 * (q0 * q2 + q1 * q3)) * (180 / math.pi)
    pitch = math.atan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * (180 / math.pi)
    # roll = math.atan2(2*(q0*q3-q1*q2), q0*q0+q1*q1-q2*q2-q3*q3)*(180/math.pi)
    euler = [yaw, pitch]

    return euler

def get_faces(image, threshold=0.5, minsize=20):
    # img = img_to_np(image)
    # face detection parameters
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    faces = []

    bounding_boxes, points = detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    idx = 0
    for bb in bounding_boxes:
        # print(bb[:4])
        # img = image.crop(bb[:4])
        # bb[2:4] -= bb[:2]
        # faces.append(Face(*bb, img))

        landmark = points[:, idx].reshape((2, 5)).T
        bbox = bb[0:4]
        euler = calculate_euler(image, landmark)
        # print(landmark)
        # test_img = image[...,::-1]
        # for i in range(np.shape(landmark)[0]):
        #     x = int(landmark[i][0])
        #     y = int(landmark[i][1])
        #     cv2.circle(test_img, (x, y), 2, (255, 0, 0))
        #
        # img_size = np.asarray(image.shape)[0:2]
        # bb[0] = np.maximum(bb[0] - face_crop_margin / 2, 0)
        # bb[1] = np.maximum(bb[1] - face_crop_margin / 2, 0)
        # bb[2] = np.minimum(bb[2] + face_crop_margin / 2, img_size[1])
        # bb[3] = np.minimum(bb[3] + face_crop_margin / 2, img_size[0])
        # cropped = image[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]), :]
        # img = misc.imresize(cropped, (face_crop_size, face_crop_size), interp='bilinear')
        img = alignment(image, bbox, landmark, (112,112))
        if face_crop_size != 112:
            img = misc.imresize(img, (face_crop_size, face_crop_size), interp='bilinear')

        faces.append(Face(bb[0], bb[1], bb[2], bb[3], bb[4], img.copy(), euler))
        idx = idx + 1
    # plt.imshow(test_img)
    # plt.show()
    return faces
