'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Lab Sheet 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math

'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

# print("here", flush=True)
if __name__ == '__main__': 
    bDisplayAxis = True
    img_width = 640
    img_height = 480

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    name_list.append('sphere_r')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, -4],
                     [0, 1, 0, 2],
                     [0, 0, 1, -2],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    name_list.append('sphere_g')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, -7],
                     [0, 1, 0, 2],
                     [0, 0, 1, 3],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    name_list.append('sphere_b')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, 4],
                     [0, 1, 0, 1.5],
                     [0, 0, 1, 4],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if bDisplayAxis:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * 45*5/180.
    # theta = 0.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * 80/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * 45*5/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]


    
    #########################################
    '''
    Question 1: Epipolar line
    Hint: check reference here
    http://www.open3d.org/docs/0.7.0/tutorial/Basic/visualization.html#draw-line-set

    Write your code here
    '''

    pt0 = np.array([0, 20/math.sqrt(2), 20/math.sqrt(2), 1.])   # camera 0 centre
    pt1 = H_list[-1][:, 3]                                      # last sphere
    end_pts = [pt0[:3], pt1[:3]]
    lines = [[0, 1]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(end_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    obj_meshes.append(line_set)
    #########################################


    # set camera intrinsics
    K = o3d.camera.PinholeCameraIntrinsic(640, 480, 415.69219381653056, 415.69219381653056, 319.5, 239.5)
    # print(K)
    # print(K.intrinsic_matrix.shape)
    print('Pose_0\n', H0_wc)
    print('Pose_1\n', H1_wc)
    print('Intrinsics\n', K.intrinsic_matrix)
    # o3d.io.write_pinhole_camera_intrinsic("test.json", K)


    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()


    #########################################
    '''
    Question 2: Extend epipolar line in the image
    
    Write your code here
    '''

    # world to camera coordinate transform
    pts_w = np.stack((pt0, pt1))
    # camera to image coornidate transform
    pts_cam1 = transform_points(pts_w, H1_wc)
    pts_img1 = []
    # print(K.intrinsic_matrix.shape)
    for i in range(pts_cam1.shape[0]):
        img_pt = np.matmul(K.intrinsic_matrix, pts_cam1[i, :].reshape(3, 1)).reshape(3,)
        img_pt /= img_pt[2]
        # print(img_pt)
        pts_img1.append(img_pt[:2])
    # get line end points across the entire image
    cam0_centre = pts_img1[0]
    sphere_centre = pts_img1[1]
    left_y = sphere_centre[1] + ((0-sphere_centre[0]) * (sphere_centre[1]-cam0_centre[1]) / (sphere_centre[0]-cam0_centre[0]))
    left_end_pt = np.array([0, left_y]).astype(int)
    right_x = sphere_centre[0] + ((0-sphere_centre[1]) * (sphere_centre[0]-cam0_centre[0]) / (sphere_centre[1]-cam0_centre[1]))
    right_end_pt = np.array([right_x, 0]).astype(int)
    # draw the pts and line
    img = cv2.imread('view1.png')
    img = cv2.circle(img, sphere_centre.astype(int), radius=0, color=(0, 0, 255), thickness=4)
    img = cv2.circle(img, left_end_pt, radius=0, color=(255, 0, 0), thickness=4)
    img = cv2.circle(img, right_end_pt, radius=0, color=(0, 255, 0), thickness=4)
    img = cv2.line(img, left_end_pt, right_end_pt, (0,255,0), 1)
    cv2.imwrite('view1_eline_extend.png', img)
    #########################################


    #########################################
    '''
    Question 3: Arbitrary epipolar line

    Write your code here
    '''

    # compute essential and fundamental matrix
    # relative pose from cam1 to cam0
    H_10 = np.matmul(H0_wc, np.linalg.inv(H1_wc))
    # print(H_10)
    Rot = H_10[:3, :3].T
    trans = H_10[:3, 3]
    tx = np.array([
        [0, -trans[2], trans[1]],
        [trans[2], 0, -trans[0]],
        [-trans[1], trans[0], 0]
    ])
    E_mat = np.matmul(Rot, tx)
    print('Essential Matrix')
    print(E_mat)
    Kinv = np.linalg.inv(K.intrinsic_matrix)
    F_mat = np.matmul(np.matmul(Kinv.T, E_mat), Kinv)
    print('Fundamental Matrix:')
    print(F_mat)
    print('rank:', np.linalg.matrix_rank(F_mat))

    # get an image point in camera 0
    pt1_cam0 = transform_points(pt1.reshape(1, 4), H0_wc)
    pt1_img0 = np.matmul(K.intrinsic_matrix, pt1_cam0.reshape(3, 1))
    pt1_img0 /= pt1_img0[2]

    # compute epipolar line with the image point
    def compute_epipolar_lines(pts, F_mat, img_idx):
        '''
        pts: (3xN) matrix of homogeneous coordinates
        F_mat: (3x3) fundamental matrix
        l: (3xN) matrix with each column being a line by a, b, c
        '''
        # print('computing epipolar lines')
        # points from first image (left)
        if img_idx == 1:
            l = np.matmul(F_mat, pts)
        # points from second image (right)
        elif img_idx == 2:
            l = np.matmul(F_mat.T, pts)
        else:
            raise ValueError('Wrong image index, set 1 for left image and 2 for right image.')
        # print(l.T)
        # normalisation
        _, N = l.shape
        l_norm = []
        for i in range(N):
            norm = (l[0,i]**2 + l[1,i]**2)**0.5
            l_norm.append( l[:,i]/norm )
        l_norm = np.array(l_norm)
        # print(l_norm)
        return l_norm.T

    line0_img1 = compute_epipolar_lines(pt1_img0, F_mat, 1)
    # get the line end pts
    x0, y0 = map(int, [0, -line0_img1[2]/line0_img1[1] ])
    x1, y1 = map(int, [img_width, -(line0_img1[2]+line0_img1[0]*img_width)/line0_img1[1] ])
    # visualise
    img = cv2.imread('view1_eline_extend.png')
    img = cv2.line(img, left_end_pt, right_end_pt, (255,0,0), 1)
    cv2.imwrite('view1_eline_fmat.png', img)
    #########################################
