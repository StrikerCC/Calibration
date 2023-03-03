import matplotlib.pyplot as plt
import numpy as np
import csv

from main import test_filter_pts_add_noise_shuffle_pts


def draw_2d_table(x, y, z, z_name='', title=''):
    fig = plt.figure()
    ax1 = plt.axes()

    x, y = np.meshgrid(x, y)
    ax1.pcolor(x, y, z)

    for x, y, z in zip(x.flatten(), y.flatten(),
                       z.flatten()):
        anno = round(z, 1)
        ax1.text(x, y, anno, color='red', fontsize=25)

    ax1.set_title(title)
    ax1.set_xlabel('noise scale /pixel')
    ax1.set_ylabel('number of noisy cross marker /count')
    ax1.set_xlim(0, 11)
    ax1.set_xticks(np.arange(0, 11, 0.5))
    ax1.set_ylim(1, 5)
    ax1.set_yticks(np.arange(1, 5, 1))

    # plt
    # plt.show()

    fig.set_figwidth(20)
    fig.set_figheight(10)
    plt.savefig('./tmp/' + title + '.png')

    return


def draw_3d(x, y, z, z_name='', title=''):
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')

    x, y = np.meshgrid(x, y)
    # ax1.plot_wireframe(x, y, z)
    ax1.plot_surface(x, y, z)

    for x, y, z in zip(x.flatten(), y.flatten(),
                       z.flatten()):
        anno = round(z, 1)
        ax1.text(x, y, z, anno, color='red')

    ax1.set_title(title)
    ax1.set_xlabel('noise scale /pixel')
    ax1.set_ylabel('number of noisy cross marker /count')
    ax1.set_zlabel(z_name)
    ax1.set_xlim(0, 11)
    ax1.set_xticks(np.arange(0, 11, 0.5))
    ax1.set_ylim(1, 5)
    ax1.set_yticks(np.arange(1, 5, 1))

    plt.savefig('./tmp/' + title + '.png')
    return


def draw_table(x, y, z, z_name='', title=''):
    fields = [[0] + [ele_x for ele_x in x]]
    rows = []
    for i_y, ele_y in enumerate(y):
        rows.append([ele_y])
        for i_x, ele_x in enumerate(x):
            ele_z_str = z[i_y][i_x]
            rows[i_y].append(ele_z_str)
    rows = fields + rows
    # print(fields)
    # print(rows)
    rows = np.asarray(rows)
    np.savetxt('./tmp/' + title + '.csv', rows, fmt='%1.2f', delimiter=', ')
    # np.savetxt('./tmp/' + title + '.csv', np.zeros(3), delimiter=' ')
    return


# def vis_calib_statistics():
#     num_noisy_pts_list = [i for i in range(1, 6)]
#     # scales = [i*0.1 for i in range(50)]
#     scales = [0.5, 1, 3, 10]
#
#     # use all
#     # code 0 result:
#     pass_ratios = np.asarray([[1., 1., 1., 0.6],
#                               [1., 1., 1., 0.],
#                               [1., 1., 0.6, 0.],
#                               [1., 1., 0.4, 0.],
#                               [1., 1., 0.2, 0.]])
#     errors = np.asarray([[1.52122919, 2.72073158, 6.5950937, 7.08277504],
#                          [1.7689336, 3.96109133, 8.83312115, 11.],
#                          [2.46266971, 5.1902844, 8.87732476, 11.],
#                          [2.70886426, 5.74920459, 8.87752015, 11.],
#                          [3.66206716, 7.04464582, 9.46790161, 11.], ])
#     fitness_ratios = np.asarray([[0.96, 0.96, 0.9, 0.94],
#                                  [0.97, 0.95, 0.98, 0.],
#                                  [0.97, 0.96, 0.93, 0.],
#                                  [0.98, 0.98, 0.98, 0.],
#                                  [0.96, 0.99, 0.97, 0.], ])
#     setting = 'use 14 vertical, 24 horizontal'
#     draw_3d(scales, num_noisy_pts_list, pass_ratios, 'pass_ratios /%', setting + ' calibration pass ratio')
#     draw_3d(scales, num_noisy_pts_list, errors, 'errors /mm', setting + ' validation errors')
#     draw_3d(scales, num_noisy_pts_list, fitness_ratios, 'fitness /%', setting + ' fitness')
#
#     # use 7 vertical, 24 horizontal
#     # code 1 result:
#     pass_ratios = np.asarray([[1., 1., 1., 0.6],
#                               [1., 1., 0.8, 0.],
#                               [1., 1., 0.8, 0.],
#                               [1., 1., 0.2, 0.],
#                               [1., 1., 0., 0.]])
#     errors = np.asarray([[1.41502662, 2.00859615, 5.57448029, 8.47087996],
#                          [1.93235949, 3.56727006, 7.54159067, 11.],
#                          [2.38159442, 5.30750277, 9.02503175, 11.],
#                          [3.03801229, 7.29629751, 9.66623373, 11.],
#                          [4.02369639, 7.59743446, 11., 11.], ])
#     fitness_ratios = np.asarray([[0.89, 0.9, 0.85, 0.89],
#                                  [0.89, 0.93, 0.91, 0.],
#                                  [0.87, 0.85, 0.91, 0.],
#                                  [0.91, 0.91, 0.9, 0.],
#                                  [0.92, 0.92, 0., 0.], ])
#     setting = 'use 7 vertical, 24 horizontal'
#     draw_3d(scales, num_noisy_pts_list, pass_ratios, 'pass_ratios /%', setting + ' calibration pass ratio')
#     draw_3d(scales, num_noisy_pts_list, errors, 'errors /mm', setting + ' validation errors')
#     draw_3d(scales, num_noisy_pts_list, fitness_ratios, 'fitness /%', setting + ' fitness')
#
#     # use 7 vertical, 12 horizontal
#     # code 2 result:
#     pass_ratios = np.asarray([[1., 1., 1., 0.6],
#                               [1., 1., 0.6, 0.],
#                               [1., 1., 0.2, 0.],
#                               [1., 1., 0.2, 0.],
#                               [1., 1., 0., 0.]])
#     errors = np.asarray([[4.30367374, 4.06234935, 7.57905884, 8.39313788],
#                          [5.205105, 5.21995961, 8.68687468, 11.],
#                          [3.94775711, 7.3719555, 8.33927008, 11.],
#                          [5.24997517, 8.04687751, 8.17451477, 11.],
#                          [5.94087041, 9.14104277, 11., 11.], ])
#     fitness_ratios = np.asarray([[1., 1., 1., 1.],
#                                  [1., 1., 1., 0.],
#                                  [1., 1., 1., 0.],
#                                  [1., 0.99, 1., 0.],
#                                  [1., 0.99, 0., 0.], ])
#     setting = 'use 7 vertical, 12 horizontal'
#     draw_3d(scales, num_noisy_pts_list, pass_ratios, 'pass_ratios /%', setting + ' calibration pass ratio')
#     draw_3d(scales, num_noisy_pts_list, errors, 'errors /mm', setting + ' validation errors')
#     draw_3d(scales, num_noisy_pts_list, fitness_ratios, 'fitness /%', setting + ' fitness')
#
#     # use 7 vertical, 8 horizontal
#     # code 3 result:
#     pass_ratios = np.asarray([[1., 1., 1., 0.2],
#                               [1., 1., 0.8, 0.],
#                               [1., 1., 0.4, 0.],
#                               [1., 1., 0., 0.],
#                               [1., 1., 0., 0.]])
#     errors = np.asarray([[4.57656078, 5.69735683, 7.09537322, 5.80400081],
#                          [6.11373689, 6.03675609, 9.35061058, 11.],
#                          [4.46819111, 7.12269279, 9.68313746, 11.],
#                          [4.90489307, 7.29966117, 11., 11.],
#                          [6.53459105, 8.74481057, 11., 11.], ])
#     fitness_ratios = np.asarray([[0.99, 0.96, 1., 1.],
#                                  [1., 0.99, 0.84, 0.],
#                                  [1., 0.99, 0.91, 0.],
#                                  [0.99, 0.98, 0., 0.],
#                                  [0.99, 0.91, 0., 0.], ])
#     setting = 'use 7 vertical, 8 horizontal'
#     draw_3d(scales, num_noisy_pts_list, pass_ratios, 'pass_ratios /%', setting + ' calibration pass ratio')
#     draw_3d(scales, num_noisy_pts_list, errors, 'errors /mm', setting + ' validation errors')
#     draw_3d(scales, num_noisy_pts_list, fitness_ratios, 'fitness /%', setting + ' fitness')
#
#     # use 14 vertical, 12 horizontal
#     # code 4 result:
#     pass_ratios = np.asarray([[1., 1., 1., 0.4],
#                               [1., 1., 0.4, 0.],
#                               [1., 1., 0.4, 0.],
#                               [1., 1., 0., 0.],
#                               [1., 1., 0., 0.]])
#     errors = np.asarray([[3.54867153, 4.7891182, 7.68710174, 6.99215221],
#                          [4.18687822, 4.45267961, 8.19004236, 11.],
#                          [4.92338523, 5.58478728, 7.46751867, 11.],
#                          [4.58275082, 7.89869362, 11., 11.],
#                          [4.25850618, 6.38891282, 11., 11.], ])
#     fitness_ratios = np.asarray([[1., 1., 0.99, 1.],
#                                  [1., 1., 1., 0.],
#                                  [1., 0.99, 1., 0.],
#                                  [1., 1., 0., 0.],
#                                  [1., 1., 0., 0.], ])
#     setting = 'use 14 vertical, 12 horizontal'
#     draw_3d(scales, num_noisy_pts_list, pass_ratios, 'pass_ratios /%', setting + ' calibration pass ratio')
#     draw_3d(scales, num_noisy_pts_list, errors, 'errors /mm', setting + ' validation errors')
#     draw_3d(scales, num_noisy_pts_list, fitness_ratios, 'fitness /%', setting + ' fitness')
#
#     # use 14 vertical, 8 horizontal
#     # code 4 result:
#     pass_ratios = np.asarray([[1., 1., 1., 0.2],
#                               [1., 1., 1., 0.],
#                               [1., 1., 0.2, 0.],
#                               [1., 1., 0., 0.],
#                               [1., 1., 0., 0.], ])
#     errors = np.asarray([[3.92252607, 4.57870336, 8.13249111, 9.28944947],
#                          [4.00331667, 5.83079576, 9.42769306, 11.],
#                          [4.50129409, 6.80930401, 7.67712225, 11.],
#                          [5.2950468, 7.05810613, 11., 11.],
#                          [4.86405829, 7.38496426, 11., 11.], ])
#     fitness_ratios = np.asarray([[0.83, 0.82, 0.89, 0.7],
#                                  [0.9, 0.83, 0.78, 0.],
#                                  [0.89, 0.89, 0.8, 0.],
#                                  [0.85, 0.78, 0., 0.],
#                                  [0.83, 0.83, 0., 0.], ])
#     setting = 'use 14 vertical, 8 horizontal'
#     draw_3d(scales, num_noisy_pts_list, pass_ratios, 'pass_ratios /%', setting + ' calibration pass ratio')
#     draw_3d(scales, num_noisy_pts_list, errors, 'errors /mm', setting + ' validation errors')
#     draw_3d(scales, num_noisy_pts_list, fitness_ratios, 'fitness /%', setting + ' fitness')

def vis_calib_statistics(scales, num_noisy_pts_list, name_2_results):
    for protocol_name, result_table in name_2_results.items():
        pass_ratios = result_table['pass_ratios']
        errors = result_table['errors_max']
        fitness_ratios = result_table['fitness_ratios_min']

        setting = protocol_name
        draw_table(scales, num_noisy_pts_list, pass_ratios, 'pass_ratios /%', setting + ' calibration pass ratio')
        draw_table(scales, num_noisy_pts_list, errors, 'errors /mm', setting + ' validation errors')
        draw_table(scales, num_noisy_pts_list, fitness_ratios, 'fitness /%', setting + ' fitness')


def main():
    # x = [i for i in range(5)]
    # y = [i for i in range(5, 10)]
    #
    # z = np.asarray([[ele_x + ele_y for ele_x in x] for ele_y in y])
    # draw_3d(x, y, z)
    scales, num_noisy_pts_list, name_2_results = test_filter_pts_add_noise_shuffle_pts()
    vis_calib_statistics(scales, num_noisy_pts_list, name_2_results)
    return


if __name__ == '__main__':
    main()
