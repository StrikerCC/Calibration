import json
import os
import random

import slam_lib.format
import slam_lib.mapping

import cv2
import numpy as np
import open3d as o3

import utils


class Calib:
    def __init__(self):
        self.verbose_print = False

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

        self.pts_img_train = None
        self.pts_world_train = None

        self.pts_img_varify = None
        self.pts_world_varify = None

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

        if self.verbose_print:
            print('rgb paths: ', rgb_paths)
            print('depth paths: ', rgb_paths)

        for rgb_file_path in rgb_paths:
            self.calib_rgb_image_paths.append(rgb_imgs_folder_path + '/' + rgb_file_path)
        for depth_file_path in depth_paths:
            self.calib_depth_image_paths.append(depth_imgs_folder_path + '/' + depth_file_path)

        return

    def extract_marker(self, world_pts_file_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/readings.txt',
                       world_pts_front_file_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/readings_front.txt',
                       rgb_imgs_folder_path='./data/UVisionAdjust_Picture_2022-07-22-14-57-42-559/pts.json',
                       num_train=10, num_varify=5):
        extracted_marker_pts_img_coord = []
        extracted_marker_pts_front_img_coord = []

        target_marker_pts_in_world_frame = []
        target_marker_pts_front_in_world_frame = []

        pts_2_img_id = []
        pts_front_2_img_id = []

        # TODO: model training

        '''load extracted pt'''
        f = open(rgb_imgs_folder_path, 'r', encoding='utf-8')
        lines = f.read()
        if lines.startswith(u'\ufeff'):
            lines = lines.encode('utf8')[3:].decode('utf8')

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

        '''reformat target marker pts'''
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

        if self.verbose_print:
            print('front points ', len(extracted_marker_pts_front_img_coord))
            print(extracted_marker_pts_front_img_coord)
            print('rear points ', len(extracted_marker_pts_img_coord))
            print(extracted_marker_pts_img_coord)
        return

    def split_train_var_set(self, num_varify=5):
        self.pts_world_train = self.pts_world[:-num_varify]
        self.pts_world_varify = self.pts_world[-num_varify:]

        self.pts_img_train = self.pts_img[:-num_varify]
        self.pts_img_varify = self.pts_img[-num_varify:]

        if self.verbose_print:
            print('training points ', len(self.pts_img_train))
            print(self.pts_world_train)
            print(self.pts_img_train)
            print('validating points ', len(self.pts_img_varify))
            print(self.pts_world_varify)
            print(self.pts_img_varify)

        return

    def calibrate(self, num_repeat=120):
        """"""
        assert len(self.pts_world_train) == len(self.pts_img_train)

        pass_ratio = 0
        extracted_marker_pts_virtual_3d_dis_error_best = 99999
        cam_pose_best = np.eye(4)

        for i in range(num_repeat):

            # shuffle world and image points
            id_shuffled = [i for i in range(len(self.pts_world_train))]
            random.shuffle(id_shuffled)
            target_marker_pts_in_world_frame_train = self.pts_world_train[id_shuffled]
            extracted_marker_pts_img_coord_train = self.pts_img_train[id_shuffled]

            if self.verbose_print:
                print('calibration #', i)
                print('target_marker_pts_in_world_frame_train', len(target_marker_pts_in_world_frame_train))
                print(target_marker_pts_in_world_frame_train)
                print('extracted_marker_pts_img_coord_train', len(extracted_marker_pts_img_coord_train))
                print(extracted_marker_pts_img_coord_train)

            '''extract marker pts from image'''
            # self.extract_marker()

            '''set verification pts'''
            # train_pts_len = int(len(self.pts_world_train) * train_ratio)

            '''solve camera pose'''
            # _, rotation_vector_cam_2_world, transl_cam_2_world = cv2.solvePnP(target_marker_pts_in_world_frame_train, extracted_marker_pts_img_coord_train,
            #                                    self.intrinsic, self.dist_coefficient)
            _, rotation_vector_cam_2_world, transl_cam_2_world, inlines = cv2.solvePnPRansac(
                target_marker_pts_in_world_frame_train, extracted_marker_pts_img_coord_train,
                self.intrinsic, self.dist_coefficient)
            if self.verbose_print:
                print('inlines', inlines)

            rotation_cam_2_world, _ = cv2.Rodrigues(rotation_vector_cam_2_world)
            self.cam_pose = slam_lib.mapping.rt_2_tf(rotation_cam_2_world.T,
                                                     -np.matmul(rotation_cam_2_world.T, transl_cam_2_world))

            if self.verbose_print:
                print('rotation_cam_2_world:\n', rotation_cam_2_world.reshape((-1, 1)))
                print('transl_cam_2_world\n', transl_cam_2_world)
            '''compute error'''
            extracted_marker_pts_virtual_3d_dis_error, extracted_marker_pts_depth_3d_dis_error, target_marker_pts_reproject_dis = self.compute_error()

            if self.verbose_print:
                print('extracted_marker_pts_virtual_3d_dis_error\n', extracted_marker_pts_virtual_3d_dis_error, '\n',
                      np.mean(extracted_marker_pts_virtual_3d_dis_error))
                print('extracted_marker_pts_depth_3d_dis_error\n', extracted_marker_pts_depth_3d_dis_error, '\n',
                      np.mean(extracted_marker_pts_depth_3d_dis_error))
                print('target_marker_pts_reproject_dis\n', target_marker_pts_reproject_dis, '\n',
                      np.mean(target_marker_pts_reproject_dis))
            if np.mean(extracted_marker_pts_virtual_3d_dis_error) < 10:
                pass_ratio += 1 / num_repeat
            if np.mean(extracted_marker_pts_virtual_3d_dis_error_best) > np.mean(
                    extracted_marker_pts_virtual_3d_dis_error):
                extracted_marker_pts_virtual_3d_dis_error_best = extracted_marker_pts_virtual_3d_dis_error
                cam_pose_best = self.cam_pose

        self.cam_pose = cam_pose_best

        return cam_pose_best, extracted_marker_pts_virtual_3d_dis_error_best, pass_ratio

    def estimate_extracted_marker_3d(self):
        """"""
        '''recover extracted marker pts 3d coord'''
        extracted_marker_pts_normalized_cam_frame = cv2.undistortPoints(self.pts_img_varify,
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
                                                                       self.pts_world_varify)

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
            pt = [(u - self.intrinsic[0, 2]) / self.intrinsic[0, 0] * pt_depth,
                  (v - self.intrinsic[1, 2]) / self.intrinsic[1, 1] * pt_depth, pt_depth]
            pts_depth.append(pt)

        pts_depth = np.asarray(pts_depth).astype(np.float64)
        pts_depth = slam_lib.mapping.transform_pt_3d(self.cam_pose, pts_depth)

        if self.verbose_print:
            print('pts_depth', len(pts_depth))
            print(pts_depth.astype(int))
        return pts_depth

    def compute_error(self):
        """"""
        extracted_marker_pts_3d_world_frame = self.estimate_extracted_marker_3d()
        extracted_marker_pts_3d_from_depth = self.get_extracted_marker_3d_from_depth()
        target_marker_pts_in_img_frame = self.reproject_extracted_marker_3d()

        extracted_marker_pts_virtual_3d_dis_error = np.linalg.norm(
            extracted_marker_pts_3d_world_frame - self.pts_world_varify, axis=1)

        extracted_marker_pts_depth_3d_dis_error = np.linalg.norm(
            extracted_marker_pts_3d_from_depth - self.pts_world, axis=1)

        target_marker_pts_reproject_dis = np.linalg.norm(
            target_marker_pts_in_img_frame - self.pts_img, axis=1)

        # if self.verbose_print:

        return extracted_marker_pts_virtual_3d_dis_error, extracted_marker_pts_depth_3d_dis_error, target_marker_pts_reproject_dis

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
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name, rgb)
                cv2.waitKey(0)
                break

            '''cam pose'''
            cam_frame = o3.geometry.TriangleMesh().create_coordinate_frame(size=250)
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

            # o3.io.write_point_cloud('./pc_cam_00.ply', pc_cam)

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


def filter_calib_pts(calib, code_2_pts_len, code=0):
    ids_keep = []
    ids = [i for i in range(len(calib.pts_world))]
    if code == 0:  # use all
        ids_keep = ids
    elif code == 1:  # use 7 vertical, 24 horizontal
        ids_keep = ids[:14][::2]
        ids_keep += ids[14:]
    elif code == 2:  # use 7 vertical, 12 horizontal
        ids_keep = ids[:14][::2]
        ids_keep += ids[14:][::2]
    elif code == 3:  # use 7 vertical, 8 horizontal
        ids_keep = ids[:14][::2]
        ids_keep += ids[14:][::3]
    elif code == 4:  # use 14 vertical, 12 horizontal
        ids_keep = ids[:14]
        ids_keep += ids[14:][::2]
    elif code == 5:  # use 14 vertical, 8 horizontal
        ids_keep = ids[:14]
        ids_keep += ids[14:][::3]

    calib.pts_world = calib.pts_world[ids_keep]
    calib.pts_img = calib.pts_img[ids_keep]

    assert len(calib.pts_world) == code_2_pts_len[code]
    assert len(calib.pts_img) == code_2_pts_len[code]

    return


def noisy_calib_img_pts(calib, scale=1.0, num_noisy_pts=1):
    assert num_noisy_pts < len(calib.pts_img)
    pts_org = calib.pts_img

    noise = np.random.normal(scale=scale, size=pts_org.shape)
    id_shuffled = [i for i in range(len(pts_org) - num_noisy_pts)]
    random.shuffle(id_shuffled)
    noise[id_shuffled] = 0
    calib.pts_img = pts_org + noise
    return


def test_filter_pts_add_noise_shuffle_pts():
    code_2_results = {}

    code_2_protocol = {0: '14 vertical, 24 horizontal',
                       1: '7 vertical, 24 horizontal',
                       2: '7 vertical, 12 horizontal',
                       3: '7 vertical, 8 horizontal',
                       4: '14 vertical, 12 horizontal',
                       5: '14 vertical, 8 horizontal'}

    code_2_pts_len = {0: 38,
                      1: 31,
                      2: 19,
                      3: 15,
                      4: 26,
                      5: 22}

    num_random = 100
    num_noisy_pts_list = [i for i in range(1, 6)]
    # scales = [i*0.1 for i in range(50)]
    scales = [0.5, 1, 3, 10]

    for code in [0, 1, 2, 3, 4, 5]:
    # for code in [0, 5]:

        pass_ratios = np.zeros((len(num_noisy_pts_list), len(scales)))
        errors_max = np.zeros((len(num_noisy_pts_list), len(scales)))
        fitness_ratios_min = np.zeros((len(num_noisy_pts_list), len(scales)))

        print('Running test')
        print('code:', code, 'start')

        for i_noisy_pts, num_noisy_pts in enumerate(num_noisy_pts_list):
            for i_scale, scale in enumerate(scales):

                print('Calibration start')
                print('num_noisy_pts', num_noisy_pts)
                print('scale', scale)

                pass_ratio = 0
                error_highest = 11
                fitness_lowest = 0

                for i_random in range(num_random):

                    calib = Calib()
                    calib.calibrator_initialize()
                    calib.load_data(rgb_imgs_folder_path='./data/calib_data_collection_02092023/lightoff/rgb/',
                                    depth_imgs_folder_path='./data/calib_data_collection_02092023/lightoff/depth/')
                    calib.extract_marker(
                        world_pts_file_path='./data/calib_data_collection_02092023/lightoff/readings.txt',
                        rgb_imgs_folder_path='./data/calib_data_collection_02092023/lightoff/pts.json',
                        num_varify=5)

                    filter_calib_pts(calib, code_2_pts_len, code)
                    noisy_calib_img_pts(calib, scale=scale, num_noisy_pts=num_noisy_pts)

                    calib.split_train_var_set()
                    cam_pose_best, extracted_marker_pts_virtual_3d_dis_error_best, fitness = calib.calibrate()

                    # extracted_marker_pts_virtual_3d_dis_error, extracted_marker_pts_depth_3d_dis_error, target_marker_pts_reproject_dis = calib.compute_error()
                    mean_extracted_marker_pts_virtual_3d_dis_error = np.mean(
                        extracted_marker_pts_virtual_3d_dis_error_best)
                    # print('extracted_marker_pts_virtual_3d_dis_error\n', mean_extracted_marker_pts_virtual_3d_dis_error, )
                    # print('fitness\n', fitness, )

                    # rotation_cam_2_world = cam_pose_best[:3, :3].T
                    # transl_cam_2_world = np.linalg.inv(cam_pose_best)[:3, -1]
                    # print('best R:\n', rotation_cam_2_world.reshape((-1, 1)))
                    # print('best T:\n', transl_cam_2_world)
                    if mean_extracted_marker_pts_virtual_3d_dis_error < 10:
                        pass_ratio += 1 / num_random
                        if error_highest == 11 or mean_extracted_marker_pts_virtual_3d_dis_error > error_highest:
                            error_highest = mean_extracted_marker_pts_virtual_3d_dis_error
                            fitness_lowest = fitness

                pass_ratios[i_noisy_pts][i_scale] = pass_ratio
                errors_max[i_noisy_pts][i_scale] = error_highest
                fitness_ratios_min[i_noisy_pts][i_scale] = fitness_lowest

                print('pass_ratio', pass_ratio)
                print('error_highest', error_highest)
                print('fitness_lowest', fitness_lowest)

        print('code', code, 'result:')
        print(pass_ratios)
        print(errors_max)
        print(fitness_ratios_min)

        code_2_results[code_2_protocol[code]] = {'pass_ratios': pass_ratios,
                                                 'errors_max': errors_max,
                                                 'fitness_ratios_min': fitness_ratios_min}

    return scales, num_noisy_pts_list, code_2_results


def run_calib():
    print('Running test')

    calib = Calib()
    calib.calibrator_initialize()
    # calib.load_data()
    # calib.extract_marker()

    # calib.load_data(rgb_imgs_folder_path='./data/02062023_bay2/rgb/',
    #                 depth_imgs_folder_path='./data/01312023_bay3/depth/')
    # calib.extract_marker(world_pts_file_path='./data/01312023_bay3/readings.txt',
    #                      rgb_imgs_folder_path='./data/02062023_bay2/pts.json', num_varify=5)

    calib.load_data(rgb_imgs_folder_path='./data/calib_data_collection_02092023/lightoff/rgb/',
                    depth_imgs_folder_path='./data/calib_data_collection_02092023/lightoff/depth/')
    calib.extract_marker(world_pts_file_path='./data/calib_data_collection_02092023/lightoff/readings.txt',
                         rgb_imgs_folder_path='./data/calib_data_collection_02092023/lightoff/pts.json', num_varify=5)

    calib.split_train_var_set()
    cam_pose_best, extracted_marker_pts_virtual_3d_dis_error_best, fitness = calib.calibrate()

    # extracted_marker_pts_virtual_3d_dis_error, extracted_marker_pts_depth_3d_dis_error, target_marker_pts_reproject_dis = calib.compute_error()
    mean_extracted_marker_pts_virtual_3d_dis_error = np.mean(extracted_marker_pts_virtual_3d_dis_error_best)
    print('extracted_marker_pts_virtual_3d_dis_error\n', mean_extracted_marker_pts_virtual_3d_dis_error, )
    print('fitness\n', fitness, )

    rotation_cam_2_world = cam_pose_best[:3, :3].T
    transl_cam_2_world = np.linalg.inv(cam_pose_best)[:3, -1]
    print('best R:\n', rotation_cam_2_world.reshape((-1, 1)))
    print('best T:\n', transl_cam_2_world)
    return


def main():
    # test()
    test_filter_pts_add_noise_shuffle_pts()


if __name__ == '__main__':
    main()
