import json
import os

import slam_lib.format
import slam_lib.mapping

import cv2
import numpy as np
import open3d as o3


class Calib:
    def __init__(self):
        self.calib_rdb_image_paths = []
        self.calib_depth_image_paths = []
        self.intrinsic = None
        self.cam_pose = None
        self.dist_coefficient = None
        self.target_marker_pts_in_world_frame = None
        self.extracted_marker_pts_img_coord = None
        self.img_id_2_pt_id = {}
        self.log_folder = './results/'

    def calibrator_initialize(self, cam_parameter_file_path='./config/calibration.json'):
        f = open(cam_parameter_file_path, 'r')
        cam_para = json.load(f)

        self.intrinsic = np.eye(3)
        self.intrinsic[0, 0] = cam_para['cam'][0]
        self.intrinsic[1, 1] = cam_para['cam'][1]
        self.intrinsic[0, 2] = cam_para['cam'][2]
        self.intrinsic[1, 2] = cam_para['cam'][3]

        self.dist_coefficient = np.asarray(cam_para['dist'])
        return

    def load_data(self, rgb_imgs_folder_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/rgb/', world_pts_file_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/readings.txt', depth_imgs_folder_path='./data/depth/'):
        for rgb_file_path in os.listdir(rgb_imgs_folder_path):
            self.calib_rdb_image_paths.append(rgb_imgs_folder_path + '/' + rgb_file_path)
        for depth_file_path in os.listdir(depth_imgs_folder_path):
            self.calib_depth_image_paths.append(depth_imgs_folder_path + '/' + depth_file_path)
        # self.pts_world_coord = np.loadtxt(world_pts_file_path)

        self.target_marker_pts_in_world_frame = []
        f = open(world_pts_file_path, 'r')
        for line in f.readlines():
            yz = list(map(float, line.split(';')[::-1]))
            self.target_marker_pts_in_world_frame.append([0.0] + yz)     # all points in same yz plane in world frame
        self.target_marker_pts_in_world_frame = np.asarray(self.target_marker_pts_in_world_frame, dtype=np.float64)
        return

    def extract_marker(self, rgb_imgs_folder_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/pts.json'):
        self.extracted_marker_pts_img_coord = []
        # TODO: model training

        # load extracted pt
        f = open(rgb_imgs_folder_path, 'r', encoding='utf-8')
        lines = f.read()

        if lines.startswith(u'\ufeff'):
            lines = lines.encode('utf8')[3:].decode('utf8')
        print(len(lines))

        img_and_pts = json.loads(lines)
        for i_img, img_name in enumerate(img_and_pts.keys()):
            self.img_id_2_pt_id[i_img] = []
            for marker_pt in img_and_pts[img_name]:   # record each image point on this image accordingly
                self.extracted_marker_pts_img_coord.append(marker_pt)                   # image points
                self.img_id_2_pt_id[i_img].append(len(self.extracted_marker_pts_img_coord)-1)  # image id correspond to imager point

        self.extracted_marker_pts_img_coord = np.asarray(self.extracted_marker_pts_img_coord).astype(np.float64)

        # reformat target marker pts
        return

    def calibrate(self, train_ratio=0.8):
        """"""
        assert len(self.target_marker_pts_in_world_frame) == len(self.extracted_marker_pts_img_coord)

        '''extract marker pts from image'''
        self.extract_marker()

        '''set verification pts'''
        train_pts_len = int(len(self.extracted_marker_pts_img_coord) * train_ratio)

        '''solve camera pose'''
        target_marker_pts_in_world_frame_train = self.target_marker_pts_in_world_frame[:train_pts_len]
        extracted_marker_pts_img_coord_train = self.extracted_marker_pts_img_coord[:train_pts_len]
        _, rotation_vector_cam_2_world, transl_cam_2_world = cv2.solvePnP(target_marker_pts_in_world_frame_train, extracted_marker_pts_img_coord_train,
                                           self.intrinsic, self.dist_coefficient)
        rotation_cam_2_world, _ = cv2.Rodrigues(rotation_vector_cam_2_world)
        self.cam_pose = slam_lib.format.rt_2_tf(rotation_cam_2_world.T, -np.matmul(rotation_cam_2_world.T, transl_cam_2_world))

        '''compute error'''
        self.compute_error()

        return

    def estimate_extracted_marker_3d(self):
        """"""
        '''recover extracted marker pts 3d coord'''
        extracted_marker_pts_normalized_cam_frame = cv2.undistortPoints(self.extracted_marker_pts_img_coord, cameraMatrix=self.intrinsic, distCoeffs=self.dist_coefficient).squeeze(axis=1)
        extracted_marker_pts_normalized_cam_frame = slam_lib.format.pts_2d_2_3d_homo(extracted_marker_pts_normalized_cam_frame)

        '''get target marker coord in cam frame'''
        tf_world_frame_2_cam_frame = np.linalg.inv(self.cam_pose)
        print(tf_world_frame_2_cam_frame)

        target_marker_pts_cam_frame = slam_lib.mapping.transform_pt_3d(tf_world_frame_2_cam_frame, self.target_marker_pts_in_world_frame)

        '''estimate extracted marker depth and then 3d coord'''
        scale_in_x = target_marker_pts_cam_frame[:, 2] / extracted_marker_pts_normalized_cam_frame[:, 2]
        extracted_marker_pts_3d_cam_frame = extracted_marker_pts_normalized_cam_frame * scale_in_x[:, None]
        extracted_marker_pts_3d_world_frame = slam_lib.mapping.transform_pt_3d(self.cam_pose, extracted_marker_pts_3d_cam_frame)
        return extracted_marker_pts_3d_world_frame

    def reproject_extracted_marker_3d(self):
        tf_world_frame_2_cam_frame = np.linalg.inv(self.cam_pose)
        target_marker_pts_in_img_frame, _ = cv2.projectPoints(self.target_marker_pts_in_world_frame, tf_world_frame_2_cam_frame[:3, :3], tf_world_frame_2_cam_frame[:3, 3], self.intrinsic, self.dist_coefficient)
        target_marker_pts_in_img_frame = target_marker_pts_in_img_frame.squeeze(axis=1)
        return target_marker_pts_in_img_frame

    def compute_error(self):
        """"""
        extracted_marker_pts_3d_world_frame = self.estimate_extracted_marker_3d()
        target_marker_pts_in_img_frame = self.reproject_extracted_marker_3d()

        extracted_marker_pts_virtual_3d_dis_error = np.linalg.norm(extracted_marker_pts_3d_world_frame - self.target_marker_pts_in_world_frame, axis=1)
        target_marker_pts_reproject_dis = np.linalg.norm(target_marker_pts_in_img_frame - self.extracted_marker_pts_img_coord, axis=1)

        print('extracted_marker_pts_virtual_3d_dis_error', extracted_marker_pts_virtual_3d_dis_error, np.mean(extracted_marker_pts_virtual_3d_dis_error))
        print('target_marker_pts_reproject_dis', target_marker_pts_reproject_dis, np.mean(target_marker_pts_reproject_dis))

    def vis_debug(self):
        """"""
        '''rgb images'''
        for i_img in range(len(self.calib_rdb_image_paths)):
            rgb = cv2.imread(self.calib_rdb_image_paths[i_img])

            cv2.imshow('pose '+str(i_img)+': '+self.calib_rdb_image_paths[i_img], rgb)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            break

        '''object target pts'''
        world_frame = o3.geometry.TriangleMesh().create_coordinate_frame(size=500)
        pc_target_markers = o3.geometry.PointCloud()
        pc_target_markers.points = o3.utility.Vector3dVector(self.target_marker_pts_in_world_frame)
        pc_target_markers.paint_uniform_color((0, 1, 0))

        sphere = o3.geometry.TriangleMesh().create_sphere(radius=10)
        tf = np.eye(4)
        tf[:3, 3] = self.target_marker_pts_in_world_frame[0, :]
        sphere.transform(tf)

        # o3.visualization.draw_geometries([pc_markers, sphere, world_frame], 'target marker points')
        # o3.visualization.draw_geometries([pc_markers])

        if self.cam_pose is not None:
            '''reproject'''
            target_marker_pts_in_img_frame = self.reproject_extracted_marker_3d()
            for i_img in range(len(self.calib_rdb_image_paths)):
                rgb = cv2.imread(self.calib_rdb_image_paths[i_img])
                for id_pt in self.img_id_2_pt_id[i_img]:
                    target_marker_pt_in_img_frame = target_marker_pts_in_img_frame[id_pt].astype(int)
                    rgb = cv2.circle(rgb, target_marker_pt_in_img_frame, radius=5, color=(0, 0, 255), thickness=1)

                cv2.imshow('target points in pose '+str(i_img)+': '+self.calib_rdb_image_paths[i_img], rgb)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break
            '''cam pose'''
            cam_frame = o3.geometry.TriangleMesh().create_coordinate_frame(size=300)
            cam_frame.transform(self.cam_pose)

            pc_extracted_markers = o3.geometry.PointCloud()
            pc_extracted_markers.points = o3.utility.Vector3dVector(self.estimate_extracted_marker_3d())
            pc_extracted_markers.paint_uniform_color((1, 0, 0))
            o3.visualization.draw_geometries([pc_target_markers, pc_extracted_markers, sphere, world_frame, cam_frame], 'target marker points')

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
    test()

    print('Calibration start')
    calib = Calib()
    calib.calibrator_initialize()
    calib.load_data()
    calib.extract_marker()
    calib.calibrate()
    calib.vis_debug()


if __name__ == '__main__':
    main()
