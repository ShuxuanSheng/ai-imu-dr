import os
import shutil
import numpy as np
from collections import namedtuple
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from navpy import lla2ned
from collections import OrderedDict
from dataset import BaseDataset
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data
from train_torch_filter import train_filter
from utils_plot import results_filter


def launch(args):
    if args.read_data:
        args.dataset_class.read_data(args) # 读取原始kitti数据保存成.p文件
    # dataset是实例化的KITTIDataset类型的对象，属性datasets中加载了.p文件的名字 print(dataset.datasets) -> ['2011_09_26_drive_0002_extract']
    # 等同于执行dataset = KITTIDataset(args)
    # 这里不直接写成 dataset = KITTIDataset(args)是为了保持代码的灵活，可以通过修改dataset_class的值来切换不同的数据集
    dataset = args.dataset_class(args)

    if args.train_filter:
        train_filter(args, dataset)

    if args.test_filter:
        test_filter(args, dataset)

    if args.results_filter:
        results_filter(args, dataset)


class KITTIParameters(IEKF.Parameters):
    # 这里是类级别的属性，可以直接通过类名来访问，例如KITTIParameters.g，这有点类似于c++中static成员变量的用法
    # __init__函数中的属性是实例级别的属性，实例化后通过实例化的对象访问
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8

    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2

    cov_lat = 1
    cov_up = 10

    # **kwargs接收任意数量的关键字参数，其中** 表示参数是以字典的形式传递的
    def __init__(self, **kwargs):
        super(KITTIParameters, self).__init__(**kwargs)
        self.set_param_attr()

    # set_param_attr函数的作用是当将类级别的属性绑定到每一个实例上，
    # 这么做的作用是可以为每一个不同的实例设置不同的属性值，而不是让所有实例共享同一个类级别的属性值
    # 同时，自动设置类级别属性到实例，可以避免在每个实例化对象时重复设置这些属性
    # 同时，将模型的超参数或配置选项直接传递给模型实例，而无需显式地在每个实例化过程中手动设置这些参数。
    def set_param_attr(self):
        # 获取 KITTIParameters 类的所有属性名（不包括特殊方法和属性）
        # dir(KITTIParameters) 返回 KITTIParameters 类的所有属性和方法名的列表
        # 列表推导式用于从一个可迭代对象中创建新的列表，语法是[expression for item in iterable if condition]
        attr_list = [a for a in dir(KITTIParameters) if
                     not a.startswith('__') and not callable(getattr(KITTIParameters, a))]
        # 把
        for attr in attr_list:
            setattr(self, attr, getattr(KITTIParameters, attr))


class KITTIDataset(BaseDataset):
    # OxtsData是Kitti数据集中的一部分，它提供了GPS/惯性导航系统的数据
    # 定一个命名元组类型数据，
    # 语法上，namedtuple接收两个参数：typename(这里是OxtsPacket)、一个可迭代对象例如列表
    # 这样可以通过OxtsPacket.lat这种方式更方便的访问数据
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, '
                            + 'roll, pitch, yaw, '
                            + 'vn, ve, vf, vl, vu, '
                            + 'ax, ay, az, af, al, au, '
                            + 'wx, wy, wz, wf, wl, wu, '
                            + 'pos_accuracy, vel_accuracy, '
                            + 'navstat, numsats, '
                            + 'posmode, velmode, orimode')

    # Bundle into an easy-to-access structure
    OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
    min_seq_dim = 25 * 100  # 60 s
    # 有问题的数据，后续需要从数据集中移除
    datasets_fake = ['2011_09_26_drive_0093_extract', '2011_09_28_drive_0039_extract',
                     '2011_09_28_drive_0002_extract']
    """
    '2011_09_30_drive_0028_extract' has trouble at N = [6000, 14000] -> test data
    '2011_10_03_drive_0027_extract' has trouble at N = 29481
    '2011_10_03_drive_0034_extract' has trouble at N = [33500, 34000]
    """

    # training set to the raw data of the KITTI dataset.
    # The following dict lists the name and end frame of each sequence that
    # has been used to extract the visual odometry / SLAM training set
    odometry_benchmark = OrderedDict()
    odometry_benchmark["2011_09_30_drive_0020_extract"] = [0, 11347]

    """
    odometry_benchmark["2011_10_03_drive_0027_extract"] = [0, 45692]
    odometry_benchmark["2011_10_03_drive_0042_extract"] = [0, 12180]
    odometry_benchmark["2011_10_03_drive_0034_extract"] = [0, 47935]
    odometry_benchmark["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark["2011_09_30_drive_0016_extract"] = [0, 2950]
    odometry_benchmark["2011_09_30_drive_0018_extract"] = [0, 28659]
    odometry_benchmark["2011_09_30_drive_0020_extract"] = [0, 11347]
    odometry_benchmark["2011_09_30_drive_0027_extract"] = [0, 11545]
    odometry_benchmark["2011_09_30_drive_0028_extract"] = [11231, 53650]
    odometry_benchmark["2011_09_30_drive_0033_extract"] = [0, 16589]
    odometry_benchmark["2011_09_30_drive_0034_extract"] = [0, 12744]

    odometry_benchmark_img = OrderedDict()
    odometry_benchmark_img["2011_10_03_drive_0027_extract"] = [0, 45400]
    odometry_benchmark_img["2011_10_03_drive_0042_extract"] = [0, 11000]
    odometry_benchmark_img["2011_10_03_drive_0034_extract"] = [0, 46600]
    odometry_benchmark_img["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark_img["2011_09_30_drive_0016_extract"] = [0, 2700]
    odometry_benchmark_img["2011_09_30_drive_0018_extract"] = [0, 27600]
    odometry_benchmark_img["2011_09_30_drive_0020_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0027_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0028_extract"] = [11000, 51700]
    odometry_benchmark_img["2011_09_30_drive_0033_extract"] = [0, 15900]
    odometry_benchmark_img["2011_09_30_drive_0034_extract"] = [0, 12000]
    """

    def __init__(self, args):
        #super方法保证在执行构造函数之前先执行父类BaseDataset的构造函数
        super(KITTIDataset, self).__init__(args)

        self.datasets_validatation_filter['2011_09_30_drive_0028_extract'] = [11231, 53650]
        # self.datasets_train_filter["2011_10_03_drive_0042_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0018_extract"] = [0, 15000]
        # self.datasets_train_filter["2011_09_30_drive_0020_extract"] = [0, None]
        # self.datasets_train_filter["2011_09_30_drive_0027_extract"] = [0, None]
        # self.datasets_train_filter["2011_09_30_drive_0033_extract"] = [0, None]
        self.datasets_train_filter["2011_10_03_drive_0027_extract"] = [0, 18000]
        self.datasets_train_filter["2011_10_03_drive_0034_extract"] = [0, 31000]
        # self.datasets_train_filter["2011_09_30_drive_0034_extract"] = [0, None]

        for dataset_fake in KITTIDataset.datasets_fake:
            if dataset_fake in self.datasets:
                self.datasets.remove(dataset_fake)
            if dataset_fake in self.datasets_train:
                self.datasets_train.remove(dataset_fake)

    """
        读取KITTI数据集中的oxts数据(gnss/imu组合定位数据)，进行数据类型转换，最后经过pickle序列化后保存成.p文件
    """
    @staticmethod   #@staticmethod显式声明静态方法，类似于c++中的static
    def read_data(args):
        """
        Read the data from the KITTI dataset

        :param args:
        :return:
        """

        print("Start read_data")
        t_tot = 0  # sum of times for the all dataset
        # 通过os.listdir函数获取指定目录中所有文件和子目录的名称列表
        date_dirs = os.listdir(args.path_data_base)
        # enumerate将列表中的每个元素和它的索引组合成一个元组
        for n_iter, date_dir in enumerate(date_dirs):
            # get access to each sequence
            path1 = os.path.join(args.path_data_base, date_dir)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue
                # read data
                # 获取指定目录 path2/oxts/data 下所有的 .txt 文件，并将它们按照文件名排序存储在 oxts_files 列表中
                oxts_files = sorted(glob.glob(os.path.join(path2, 'oxts', 'data', '*.txt')))
                oxts = KITTIDataset.load_oxts_packets_and_poses(oxts_files)

                """ Note on difference between ground truth and oxts solution:
                    - orientation is the same
                    - north and east axis are inverted
                    - position are closed to but different
                    => oxts solution is not loaded
                """

                print("\n Sequence name : " + date_dir2)
                # if len(oxts) < KITTIDataset.min_seq_dim:  #  sequence shorter than 30 s are rejected
                #     cprint("Dataset is too short ({:.2f} s)".format(len(oxts) / 100), 'yellow')
                #     continue
                lat_oxts = np.zeros(len(oxts))
                lon_oxts = np.zeros(len(oxts))
                alt_oxts = np.zeros(len(oxts))
                roll_oxts = np.zeros(len(oxts))
                pitch_oxts = np.zeros(len(oxts))
                yaw_oxts = np.zeros(len(oxts))
                roll_gt = np.zeros(len(oxts))
                pitch_gt = np.zeros(len(oxts))
                yaw_gt = np.zeros(len(oxts))
                t = KITTIDataset.load_timestamps(path2)
                acc = np.zeros((len(oxts), 3))
                acc_bis = np.zeros((len(oxts), 3))
                gyro = np.zeros((len(oxts), 3))
                gyro_bis = np.zeros((len(oxts), 3))
                p_gt = np.zeros((len(oxts), 3))
                v_gt = np.zeros((len(oxts), 3))
                v_rob_gt = np.zeros((len(oxts), 3))

                k_max = len(oxts)
                for k in range(k_max):
                    oxts_k = oxts[k]
                    t[k] = 3600 * t[k].hour + 60 * t[k].minute + t[k].second + t[k].microsecond / 1e6
                    lat_oxts[k] = oxts_k[0].lat #oxts_k数据类型是OxtsData(namedtuple)，其中第一个字段是oxts_packets
                    lon_oxts[k] = oxts_k[0].lon
                    alt_oxts[k] = oxts_k[0].alt
                    acc[k, 0] = oxts_k[0].af
                    acc[k, 1] = oxts_k[0].al
                    acc[k, 2] = oxts_k[0].au
                    acc_bis[k, 0] = oxts_k[0].ax
                    acc_bis[k, 1] = oxts_k[0].ay
                    acc_bis[k, 2] = oxts_k[0].az
                    gyro[k, 0] = oxts_k[0].wf
                    gyro[k, 1] = oxts_k[0].wl
                    gyro[k, 2] = oxts_k[0].wu
                    gyro_bis[k, 0] = oxts_k[0].wx
                    gyro_bis[k, 1] = oxts_k[0].wy
                    gyro_bis[k, 2] = oxts_k[0].wz
                    roll_oxts[k] = oxts_k[0].roll
                    pitch_oxts[k] = oxts_k[0].pitch
                    yaw_oxts[k] = oxts_k[0].yaw
                    v_gt[k, 0] = oxts_k[0].ve
                    v_gt[k, 1] = oxts_k[0].vn
                    v_gt[k, 2] = oxts_k[0].vu
                    v_rob_gt[k, 0] = oxts_k[0].vf
                    v_rob_gt[k, 1] = oxts_k[0].vl
                    v_rob_gt[k, 2] = oxts_k[0].vu
                    p_gt[k] = oxts_k[1][:3, 3]  #oxts_k数据类型是OxtsData(namedtuple)，其中第二个字段是根据lla和rpy转换成的R和T
                    Rot_gt_k = oxts_k[1][:3, :3]
                    roll_gt[k], pitch_gt[k], yaw_gt[k] = IEKF.to_rpy(Rot_gt_k)

                t0 = t[0]
                t = np.array(t) - t[0]
                # some data can have gps out
                if np.max(t[:-1] - t[1:]) > 0.1:
                    cprint(date_dir2 + " has time problem", 'yellow')
                ang_gt = np.zeros((roll_gt.shape[0], 3))
                ang_gt[:, 0] = roll_gt
                ang_gt[:, 1] = pitch_gt
                ang_gt[:, 2] = yaw_gt

                p_oxts = lla2ned(lat_oxts, lon_oxts, alt_oxts, lat_oxts[0], lon_oxts[0],
                                 alt_oxts[0], latlon_unit='deg', alt_unit='m', model='wgs84')
                p_oxts[:, [0, 1]] = p_oxts[:, [1, 0]]  # see note，oxts数据和gt东向、北向是反的

                # take correct imu measurements
                u = np.concatenate((gyro_bis, acc_bis), -1)
                # convert from numpy
                t = torch.from_numpy(t)
                p_gt = torch.from_numpy(p_gt)
                v_gt = torch.from_numpy(v_gt)
                ang_gt = torch.from_numpy(ang_gt)
                u = torch.from_numpy(u)

                # convert to float
                t = t.float()
                u = u.float()
                p_gt = p_gt.float()
                ang_gt = ang_gt.float()
                v_gt = v_gt.float()

                mondict = {
                    't': t, 'p_gt': p_gt, 'ang_gt': ang_gt, 'v_gt': v_gt,
                    'u': u, 'name': date_dir2, 't0': t0
                    }

                t_tot += t[-1] - t[0]
                KITTIDataset.dump(mondict, args.path_data_save, date_dir2)
        print("\n Total dataset duration : {:.2f} s".format(t_tot))

    @staticmethod
    def prune_unused_data(args):
        """
        Deleting image and velodyne
        Returns:

        """

        unused_list = ['image_00', 'image_01', 'image_02', 'image_03', 'velodyne_points']
        date_dirs = ['2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

        for date_dir in date_dirs:
            path1 = os.path.join(args.path_data_base, date_dir)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue
                print(path2)
                for folder in unused_list:
                    path3 = os.path.join(path2, folder)
                    if os.path.isdir(path3):
                        print(path3)
                        shutil.rmtree(path3)

    @staticmethod
    def subselect_files(files, indices):
        try:
            files = [files[i] for i in indices]
        except:
            pass
        return files

    """
        返回绕 x 轴旋转角度 t 的旋转矩阵
    """
    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet.
        """
        er = 6378137.  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = KITTIDataset.rotx(packet.roll)
        Ry = KITTIDataset.roty(packet.pitch)
        Rz = KITTIDataset.rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t

    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transformation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    @staticmethod
    def load_oxts_packets_and_poses(oxts_files):
        """Generator to read OXTS ground truth data.
           Poses are given in an East-North-Up coordinate system
           whose origin is the first GPS position.
        """
        # Scale for Mercator projection (from first lat value)
        # 墨卡托投影的比例（从第一个纬度值开始）
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []

        for filename in oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    # 后5列数据分别是navstat、numsats、posmode、velmode、orimode
                    line[:-5] = [float(x) for x in line[:-5]] #选择除了最后五个元素外的所有元素，并将它们转换为浮点数类型。
                    line[-5:] = [int(float(x)) for x in line[-5:]] #选择最后五个元素外的所有元素，并将它们转换为整数类型。

                    packet = KITTIDataset.OxtsPacket(*line) #OxtsPacket是namedtuple类型，这里是将每一行转换成OxtsPacket这种数据格式

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.)

                    R, t = KITTIDataset.pose_from_oxts_packet(packet, scale) #从lla展开成站心坐标

                    if origin is None:  #默认从第一帧gps为展开原点
                        origin = t

                    T_w_imu = KITTIDataset.transform_from_rot_trans(R, t - origin)

                    oxts.append(KITTIDataset.OxtsData(packet, T_w_imu))  #packet存放的是原始kitti格式的数据，T_w_imu是转换成局部站心坐标后从world到imu的transform矩阵
        return oxts

    @staticmethod
    def load_timestamps(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps

    def load_timestamps_img(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'image_00', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps

def test_filter(args, dataset):
    iekf = IEKF()  #初始化一个iekf实例，并设置Q矩阵的初始值
    torch_iekf = TORCHIEKF()  #初始化一个TORCHIEKF实例，并设置Q矩阵的初始值

    # put Kitti parameters
    iekf.filter_parameters = KITTIParameters()
    iekf.set_param_attr()
    torch_iekf.filter_parameters = KITTIParameters()
    torch_iekf.set_param_attr()

    torch_iekf.load(args, dataset)
    iekf.set_learned_covariance(torch_iekf)  #论文中提到滤波器在线估计的时候Q是固定的，但是Q的值是神经网络学好的，存在iekfnets.p中，加载进来
    for i in range(0, len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)
        if dataset_name not in dataset.odometry_benchmark.keys():
            continue  #odometry_benchmark中的seq是有slam的，用于对比，所以非odometry_benchmark中的seq直接跳过
        print("Test filter on sequence: " + dataset_name)
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i, to_numpy=True)
        N = None
        u_t = torch.from_numpy(u).double()
        # acc和gyr输入到torch_iekf网络中，预测
        v_gt_t = torch.from_numpy(v_gt).double()
        wheel_encoder = torch.norm(v_gt_t, dim=1)  # v_gt是enu速度
        wheel_t = wheel_encoder.unsqueeze(1)
        measurements_covs = torch_iekf.forward_nets(u_t,wheel_t)
        measurements_covs = measurements_covs.detach().numpy()
        start_time = time.time()
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs,
                                                                   v_gt, p_gt, N,
                                                                   ang_gt[0])
        diff_time = time.time() - start_time
        print("Test execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time, t[-1] - t[0]))
        mondict = {
            't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
            'Rot_c_i': Rot_c_i, 't_c_i': t_c_i,
            'measurements_covs': measurements_covs,
            }
        t_reshaped = t.flatten()
        combined = np.column_stack((t_reshaped, p))
        combined_cov = np.column_stack((t_reshaped, measurements_covs))
        # 保存为 csv 文件
        np.savetxt('output_wheel_filter.csv', combined, delimiter=',')
        np.savetxt('output_cov.csv', combined_cov, delimiter=',')
        dataset.dump(mondict, args.path_results, dataset_name + "_filter.p")  #保存ai-based imu dr的结果


class KITTIArgs():
        # path_data_base = "/media/mines/46230797-4d43-4860-9b76-ce35e699ea47/KITTI/raw"
        path_data_base = "/home/ssx/shengshuxuan/datasets/kitti/2011_09_26_drive_0002_extract/" #存放原始kitti数据集的路径
        path_data_save = "../data"   #存放.p格式的kitti数据的路径
        path_results = "../results"  #存放ai-based dr 的结果的路径
        path_temp = "../temp" #存放临时文件，如iekfnet参数

        epochs = 400  #https://blog.csdn.net/weixin_45662399/article/details/136836955#:~:text=%E5%8F%82%E8%80%83%E5%8D%9A%E6%96%87%E5%92%8C%E5%9B%BE%E7%89%87%E6%9D%A5
        seq_dim = 6000

        # training, cross-validation and test dataset
        # 指定训练集、交叉验证集、测试集
        cross_validation_sequences = ['2011_09_30_drive_0028_extract']
        test_sequences = ['2011_09_30_drive_0028_extract']
        continue_training = False

        # choose what to do
        read_data = 0
        train_filter = 0
        test_filter = 1
        results_filter = 1
        dataset_class = KITTIDataset
        parameter_class = KITTIParameters


if __name__ == '__main__':
    args = KITTIArgs()
    dataset = KITTIDataset(args)
    launch(KITTIArgs)

