import json
import os

import slam_lib.format
import slam_lib.mapping

import cv2
import numpy as np
import open3d as o3


class Calib:
    def __init__(self):
        self.calib_rgb_image_paths = []
        self.calib_depth_image_paths = []
        self.intrinsic = None
        self.cam_pose = None
        self.dist_coefficient = None

        # self.target_marker_pts_in_world_frame = None
        # self.target_marker_pts_front_in_world_frame = None
        #
        # self.extracted_marker_pts_img_coord = None
        # self.extracted_marker_pts_front_img_coord = None

        self.pts_2_img_id = None

        self.pts_img = None
        self.pts_world = None

        self.log_folder = './results/'

    def calibrator_initialize(self, cam_parameter_file_path='./config/calibration.json'):
        f = open(cam_parameter_file_path, 'r')
        cam_para = json.load(f)

        self.intrinsic = np.eye(3)
        self.intrinsic[0, 0] = cam_para['cam'][0]  # fx
        self.intrinsic[1, 1] = cam_para['cam'][1]  # fy
        self.intrinsic[0, 2] = cam_para['cam'][2]  # cx
        self.intrinsic[1, 2] = cam_para['cam'][3]  # cy

        self.dist_coefficient = np.asarray(cam_para['dist'])
        return

    def load_data(self, rgb_imgs_folder_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/rgb',
                  depth_imgs_folder_path='./data/depth/',
                  ):
        rgb_paths = sorted(os.listdir(rgb_imgs_folder_path))
        depth_paths = sorted(os.listdir(depth_imgs_folder_path))

        print('rgb paths: ', rgb_paths)
        print('depth paths: ', rgb_paths)

        for rgb_file_path in rgb_paths:
            self.calib_rgb_image_paths.append(rgb_imgs_folder_path + '/' + rgb_file_path)
        for depth_file_path in depth_paths:
            self.calib_depth_image_paths.append(depth_imgs_folder_path + '/' + depth_file_path)

        return

    def extract_marker(self, world_pts_file_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/readings.txt',
                       world_pts_front_file_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/readings_front.txt',
                       rgb_imgs_folder_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/pts.json'):
        extracted_marker_pts_img_coord = []
        extracted_marker_pts_front_img_coord = []

        target_marker_pts_in_world_frame = []
        target_marker_pts_front_in_world_frame = []

        pts_2_img_id = []
        pts_front_2_img_id = []

        # TODO: model training

        # load extracted pt
        f = open(rgb_imgs_folder_path, 'r', encoding='utf-8')
        lines = f.read()

        if lines.startswith(u'\ufeff'):
            lines = lines.encode('utf8')[3:].decode('utf8')
        print(len(lines))

        img_and_pts = json.loads(lines)

        for i_img, img_name in enumerate(img_and_pts.keys()):
            # rear point
            marker_pt = img_and_pts[img_name][0]
            extracted_marker_pts_img_coord.append(marker_pt)  # image points
            pts_2_img_id.append(i_img)  # correspond to image id

            # front point
            # if len(img_and_pts[img_name]) > 1:  # record each image point on this image accordingly
            #     marker_pt = img_and_pts[img_name][1]
            #     extracted_marker_pts_front_img_coord.append(marker_pt)  # image points
            #     pts_front_2_img_id.append(i_img)  # correspond to image id

        extracted_marker_pts_img_coord = np.asarray(extracted_marker_pts_img_coord).astype(np.float64)
        extracted_marker_pts_front_img_coord = np.asarray(extracted_marker_pts_front_img_coord).astype(
            np.float64)

        # reformat target marker pts
        f = open(world_pts_file_path, 'r')
        for line in f.readlines():
            yz = list(map(float, line.split(';')))
            target_marker_pts_in_world_frame.append([0.0] + yz)  # all points in same yz plane in world frame
        target_marker_pts_in_world_frame = np.asarray(target_marker_pts_in_world_frame, dtype=np.float64)

        f = open(world_pts_front_file_path, 'r')
        for line in f.readlines():
            yz = list(map(float, line.split(';')))
            target_marker_pts_front_in_world_frame.append([0.0] + yz)  # all points in same yz plane in world frame
        target_marker_pts_front_in_world_frame = np.asarray(target_marker_pts_front_in_world_frame,
                                                            dtype=np.float64)

        # rear
        assert len(target_marker_pts_in_world_frame) == len(extracted_marker_pts_img_coord)

        # front
        target_marker_pts_front_in_world_frame = \
            target_marker_pts_front_in_world_frame[:len(extracted_marker_pts_front_img_coord)]

        self.pts_world = target_marker_pts_in_world_frame
        if len(target_marker_pts_front_in_world_frame) > 0:
            self.pts_world = np.concatenate((self.pts_world, target_marker_pts_front_in_world_frame))

        self.pts_img = extracted_marker_pts_img_coord
        if len(extracted_marker_pts_front_img_coord) > 0:
            self.pts_img = np.concatenate((self.pts_img, extracted_marker_pts_front_img_coord))

        self.pts_2_img_id = pts_2_img_id + pts_front_2_img_id

        print('front points ', len(extracted_marker_pts_front_img_coord))
        print(extracted_marker_pts_front_img_coord)
        print('rear points ', len(extracted_marker_pts_img_coord))
        print(extracted_marker_pts_img_coord)

        return

    def calibrate(self, train_ratio=0.8):
        """"""
        assert len(self.pts_world) == len(self.pts_img)
        print('pts_img', self.pts_img)
        print('pts_world', self.pts_world)
        '''extract marker pts from image'''
        # self.extract_marker()

        '''set verification pts'''
        train_pts_len = int(len(self.pts_world) * train_ratio)

        '''solve camera pose'''
        # target_marker_pts_in_world_frame_train = self.pts_world[:train_pts_len]
        # extracted_marker_pts_img_coord_train = self.pts_img[:train_pts_len]
        target_marker_pts_in_world_frame_train = self.pts_world[::-1][:train_pts_len]
        extracted_marker_pts_img_coord_train = self.pts_img[::-1][:train_pts_len]

        print('target_marker_pts_in_world_frame_train', len(target_marker_pts_in_world_frame_train))
        print(target_marker_pts_in_world_frame_train)
        print('extracted_marker_pts_img_coord_train', len(extracted_marker_pts_img_coord_train))
        print(extracted_marker_pts_img_coord_train)
        # _, rotation_vector_cam_2_world, transl_cam_2_world = cv2.solvePnP(target_marker_pts_in_world_frame_train, extracted_marker_pts_img_coord_train,
        #                                    self.intrinsic, self.dist_coefficient)
        _, rotation_vector_cam_2_world, transl_cam_2_world, inlines = cv2.solvePnPRansac(
            target_marker_pts_in_world_frame_train, extracted_marker_pts_img_coord_train,
            self.intrinsic, self.dist_coefficient)
        print(inlines)

        rotation_cam_2_world, _ = cv2.Rodrigues(rotation_vector_cam_2_world)

        print('rotation_cam_2_world:\n', rotation_cam_2_world.reshape((-1, 1)))
        print('transl_cam_2_world\n', transl_cam_2_world)

        self.cam_pose = slam_lib.mapping.rt_2_tf(rotation_cam_2_world.T,
                                                 -np.matmul(rotation_cam_2_world.T, transl_cam_2_world))

        '''compute error'''
        self.compute_error()

        return

    def estimate_extracted_marker_3d(self):
        """"""
        '''recover extracted marker pts 3d coord'''
        extracted_marker_pts_normalized_cam_frame = cv2.undistortPoints(self.pts_img,
                                                                        cameraMatrix=self.intrinsic,
                                                                        distCoeffs=self.dist_coefficient).squeeze(
            axis=1)
        extracted_marker_pts_normalized_cam_frame = slam_lib.format.pts_2d_2_3d_homo(
            extracted_marker_pts_normalized_cam_frame)

        '''get target marker coord in cam frame'''
        tf_world_frame_2_cam_frame = np.linalg.inv(self.cam_pose)
        # print(self.cam_pose)
        # print(self.cam_pose[:3, :3].reshape(9, -1))
        # print(self.cam_pose[:3, -1])
        #
        # print(tf_world_frame_2_cam_frame)
        # print(tf_world_frame_2_cam_frame[:3, :3].reshape(9, -1))
        # print(tf_world_frame_2_cam_frame[:3, -1])

        target_marker_pts_cam_frame = slam_lib.mapping.transform_pt_3d(tf_world_frame_2_cam_frame,
                                                                       self.pts_world)

        '''estimate extracted marker depth and then 3d coord'''
        scale_in_x = target_marker_pts_cam_frame[:, 2] / extracted_marker_pts_normalized_cam_frame[:, 2]
        extracted_marker_pts_3d_cam_frame = extracted_marker_pts_normalized_cam_frame * scale_in_x[:, None]
        extracted_marker_pts_3d_world_frame = slam_lib.mapping.transform_pt_3d(self.cam_pose,
                                                                               extracted_marker_pts_3d_cam_frame)
        return extracted_marker_pts_3d_world_frame

    def reproject_extracted_marker_3d(self):
        tf_world_frame_2_cam_frame = np.linalg.inv(self.cam_pose)
        target_marker_pts_in_img_frame, _ = cv2.projectPoints(self.pts_world,
                                                              tf_world_frame_2_cam_frame[:3, :3],
                                                              tf_world_frame_2_cam_frame[:3, 3], self.intrinsic,
                                                              self.dist_coefficient)
        target_marker_pts_in_img_frame = target_marker_pts_in_img_frame.squeeze(axis=1)
        return target_marker_pts_in_img_frame

    def get_extracted_marker_3d_from_depth(self):
        pts_depth = []
        for i_pt, (pt, id_img) in enumerate(zip(self.pts_img, self.pts_2_img_id)):
            u, v = pt
            u, v = int(u), int(v)
            depth = np.asarray(o3.io.read_image(self.calib_depth_image_paths[id_img]))
            pt_depth = depth[v, u]
            pt = [(u-self.intrinsic[0, 2]) / self.intrinsic[0, 0] * pt_depth, (v-self.intrinsic[1, 2]) / self.intrinsic[1, 1] * pt_depth, pt_depth]
            pts_depth.append(pt)

        pts_depth = np.asarray(pts_depth).astype(np.float64)
        pts_depth = slam_lib.mapping.transform_pt_3d(self.cam_pose, pts_depth)

        print('pts_depth', len(pts_depth))
        print(pts_depth.astype(int))
        return pts_depth


    def compute_error(self):
        """"""
        extracted_marker_pts_3d_world_frame = self.estimate_extracted_marker_3d()
        extracted_marker_pts_3d_from_depth = self.get_extracted_marker_3d_from_depth()
        target_marker_pts_in_img_frame = self.reproject_extracted_marker_3d()

        extracted_marker_pts_virtual_3d_dis_error = np.linalg.norm(
            extracted_marker_pts_3d_world_frame - self.pts_world, axis=1)

        extracted_marker_pts_depth_3d_dis_error = np.linalg.norm(
            extracted_marker_pts_3d_from_depth - self.pts_world, axis=1)

        target_marker_pts_reproject_dis = np.linalg.norm(
            target_marker_pts_in_img_frame - self.pts_img, axis=1)

        print('extracted_marker_pts_virtual_3d_dis_error', extracted_marker_pts_virtual_3d_dis_error,
              np.mean(extracted_marker_pts_virtual_3d_dis_error))
        print('extracted_marker_pts_depth_3d_dis_error', extracted_marker_pts_depth_3d_dis_error,
              np.mean(extracted_marker_pts_depth_3d_dis_error))
        print('target_marker_pts_reproject_dis', target_marker_pts_reproject_dis,
              np.mean(target_marker_pts_reproject_dis))

    def vis_debug(self):
        """"""
        '''marker detection'''
        for i_img in range(len(self.calib_rgb_image_paths)):
            rgb = cv2.imread(self.calib_rgb_image_paths[i_img])

            for pt_id, img_id in enumerate(self.pts_2_img_id):
                if i_img == img_id:
                    pt_marker = self.pts_img[pt_id].astype(int)
                    cv2.circle(rgb, pt_marker, radius=3, color=(0, 0, 255))

            win_name = 'marker at img ' + str(i_img) + ': ' + self.calib_rgb_image_paths[i_img]
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, rgb)
            cv2.waitKey(0)
            break
        cv2.destroyAllWindows()

        '''rgb images'''
        # for i_img in range(len(self.calib_rdb_image_paths)):
        #     rgb = cv2.imread(self.calib_rdb_image_paths[i_img])
        #
        #     cv2.imshow('pose '+str(i_img)+': '+self.calib_rdb_image_paths[i_img], rgb)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #
        #     # break

        '''object target pts'''
        world_frame = o3.geometry.TriangleMesh().create_coordinate_frame(size=500)
        pc_target_markers = o3.geometry.PointCloud()
        pc_target_markers.points = o3.utility.Vector3dVector(self.pts_world)
        pc_target_markers.paint_uniform_color((0, 1, 0))

        sphere = o3.geometry.TriangleMesh().create_sphere(radius=10)
        tf = np.eye(4)
        tf[:3, 3] = self.pts_world[0, :]
        sphere.transform(tf)

        # o3.visualization.draw_geometries([pc_markers, sphere, world_frame], 'target marker points')
        # o3.visualization.draw_geometries([pc_markers])

        if self.cam_pose is not None:
            '''reproject'''
            target_marker_pts_in_img_frame = self.reproject_extracted_marker_3d()
            for i_img in range(len(self.calib_rgb_image_paths)):
                rgb = cv2.imread(self.calib_rgb_image_paths[i_img])
                for id_pt, id_img in enumerate(self.pts_2_img_id):
                    if i_img == id_img:
                        target_marker_pt_in_img_frame = target_marker_pts_in_img_frame[id_pt].astype(int)
                        rgb = cv2.circle(rgb, target_marker_pt_in_img_frame, radius=5, color=(0, 0, 255), thickness=1)

                win_name = 'target points in pose ' + str(i_img) + ': ' + self.calib_rgb_image_paths[i_img]
                cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(win_name, rgb)
                cv2.waitKey(0)
                # break

            '''cam pose'''
            cam_frame = o3.geometry.TriangleMesh().create_coordinate_frame(size=100)
            cam_frame.transform(self.cam_pose)

            depth = o3.io.read_image(self.calib_depth_image_paths[0])
            # color = o3.io.read_image(self.calib_rgb_image_paths[0])

            color_cv = cv2.imread(self.calib_rgb_image_paths[0])
            color = o3.geometry.Image(color_cv)

            rgbd = o3.geometry.RGBDImage().create_from_color_and_depth(depth=depth,
                                                                       color=color,
                                                                       depth_scale=1,
                                                                       depth_trunc=50000,
                                                                       convert_rgb_to_intensity=False)

            intrinsic = o3.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(width=848, height=480, fx=self.intrinsic[0, 0], fy=self.intrinsic[1, 1],
                                     cx=self.intrinsic[0, 2], cy=self.intrinsic[1, 2])
            # extrinsic = np.eye(4)
            extrinsic = np.linalg.inv(self.cam_pose)
            # extrinsic = self.cam_pose

            pc_cam = o3.geometry.PointCloud().create_from_rgbd_image(image=rgbd, intrinsic=intrinsic,
                                                                     extrinsic=extrinsic,
                                                                     project_valid_depth_only=False)
            # pc_cam.transform(extrinsic)

            o3.visualization.draw_geometries([world_frame, cam_frame, pc_cam], 'camera points')

            pc_extracted_markers = o3.geometry.PointCloud()
            pc_extracted_markers.points = o3.utility.Vector3dVector(self.estimate_extracted_marker_3d())
            pc_extracted_markers.paint_uniform_color((1, 0, 0))

            print('obj points green, output points red')
            o3.visualization.draw_geometries(
                [pc_target_markers, pc_extracted_markers, sphere, world_frame, cam_frame, pc_cam],
                'target marker points')

            o3.io.write_point_cloud('./pc_cam_00.ply', pc_cam)

        return

    def save_log(self):
        return


def test():
    target_pt_3d = np.arange(0, 150).astype(np.float64).reshape((-1, 3))
    target_pt_2d = np.arange(0, 100).astype(np.float64).reshape((-1, 2))
    cam_mat = np.eye(3)

    print(target_pt_3d.shape)
    print(target_pt_2d.shape)
    _ = cv2.solvePnP(target_pt_3d, target_pt_2d, cam_mat, None)


def main():
    # test()

    print('Calibration start')
    calib = Calib()
    calib.calibrator_initialize()
    # calib.load_data()
    # calib.extract_marker()

    calib.load_data(rgb_imgs_folder_path='./data/12202022/rgb/',
                    depth_imgs_folder_path='./data/12202022/depth/')
    calib.extract_marker(world_pts_file_path='./data/12202022/readings.txt',
                         rgb_imgs_folder_path='./data/12202022/pts.json')

    calib.calibrate(train_ratio=1.0)

    calib.vis_debug()


if __name__ == '__main__':
    main()
