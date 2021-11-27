import cv2
import plyfile
import numpy as np
import matplotlib.pyplot as plt
import os

from kitti_reader import DatasetReaderKITTI
from feature_tracking import FeatureTracker
from utils import drawFrameFeatures, updateTrajectoryDrawing, savePly

log_name = "gftt"
if(not os.path.exists("/home/aenesbedir/bitirme_logs/"+log_name)):
    os.mkdir("/home/aenesbedir/bitirme_logs/"+log_name)

def plot_errors(errorsx,errorsy,errorsz,frame_size):
    X = np.arange(0, len(errorsx) ,1)
    fig = plt.figure(1)
    plt.plot(X,errorsx,color='r', label='X')
    plt.plot(X,errorsy,color='g', label='Y')
    plt.plot(X,errorsz,color='b', label='Z')
    plt.xlabel("Frame Number")
    plt.ylabel("Error")
    plt.title("Errors in dimensions")
    fig.savefig('/home/aenesbedir/bitirme_logs/{}/{}.png'.format(log_name,"error"), dpi=fig.dpi)

    plt.legend()
    plt.show()

if __name__ == "__main__":                      
    tracker = FeatureTracker()
    # detector = cv2.xfeatures2d.SIFT_create()
    detector = cv2.GFTTDetector_create()
    #detector = cv2.ORB_create()
    #detector = cv2.FastFeatureDetector_create()
    dataset_reader = DatasetReaderKITTI("/home/aenesbedir/00")

    K = dataset_reader.readCameraMatrix()

    prev_points = np.empty(0)
    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    camera_rot, camera_pos = np.eye(3), np.zeros((3,1))

    plt.show()

    errors_x = []
    errors_y = []
    errors_z = []
    # Process next frames
    for frame_no in range(1, 1000):
        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_frame = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        # Feature detection & filtering
        prev_points = detector.detect(prev_frame)
        prev_points = cv2.KeyPoint_convert(sorted(prev_points, key = lambda p: p.response, reverse=True))
    
        # Feature tracking (optical flow)
        prev_points, curr_points = tracker.trackFeatures(prev_frame, curr_frame, prev_points, removeOutliers=True)
        print (len(curr_points), "features left after feature tracking.")

        # Essential matrix, pose estimation
        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points = np.array([pt for (idx, pt) in enumerate(prev_points) if mask[idx] == 1])
        curr_points = np.array([pt for (idx, pt) in enumerate(curr_points) if mask[idx] == 1])
        _, R, T, _ = cv2.recoverPose(E, curr_points, prev_points, K)
        print(len(curr_points),"features left after pose estimation.")

        # Read groundtruth translation T and absolute scale for computing trajectory
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        if kitti_scale <= 0.1:
            continue

        camera_pos = camera_pos + kitti_scale * camera_rot.dot(T)
        camera_rot = R.dot(camera_rot)

        kitti_positions.append(kitti_pos)
        track_positions.append(camera_pos)
        error = np.sqrt((kitti_pos[0]-camera_pos[0])**2+(kitti_pos[2]-camera_pos[2])**2)
        error_x = kitti_pos[0]-camera_pos[0]
        error_y = kitti_pos[2]-camera_pos[2]
        error_z = kitti_pos[1]-camera_pos[1]
        errors_x.append(error_x)
        errors_y.append(error_y)
        errors_z.append(error_z)
        print("error: ",error)
        updateTrajectoryDrawing(np.array(track_positions), np.array(kitti_positions))
        drawFrameFeatures(curr_frame, prev_points, curr_points, frame_no)

        if cv2.waitKey(1) == ord('q'):
            break
           
        prev_points, prev_frame_BGR = curr_points, curr_frame_BGR
    
    plt.savefig('/home/aenesbedir/bitirme_logs/{}/{}.png'.format(log_name,"map"))
    
    plot_errors(errors_x,errors_y,errors_z,999)

    cv2.destroyAllWindows()

