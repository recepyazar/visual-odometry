i_pos[0]-camera_pos[0])**2+(kitti_pos[1]-camera_pos[1])**2+(kitti_pos[2]-camera_pos[2])**2)
        print("error: ",error)
        updateTrajectoryDrawing(np.array(track_positions), np.array(kitti_positions))
        drawFrameFeatures(curr_frame, prev_points, curr_points, frame_no)
