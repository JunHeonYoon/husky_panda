#! /usr/bin/env python 
import numpy as np
import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import torch
# import cv2
import yaml
import time

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from std_msgs.msg import Int8MultiArray, MultiArrayDimension


class DepthImageProcessor:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('masking_depth_node', anonymous=False)

        # 변환 리스너 초기화
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # CvBridge 초기화
        self.bridge = CvBridge()

        # 카메라 파라미터 초기화
        self.camera_matrix = None

        # YAML 파일에서 구 모델 로드
        config_file_path = rospy.get_param('/masking_depth_node/config_file')
        with open(config_file_path, 'r') as file:
            self.sphere_models = yaml.safe_load(file)

        self.link_names = list(self.sphere_models.keys())

        # 카메라 정보와 Depth 이미지 Subscribe
        rospy.Subscriber('/depth/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/depth/depth', Image, self.depth_image_callback)

        # 로봇에 해당하는 이미지를 masking한 Depth image와 Point clouds Publish
        self.masked_depth_pub = rospy.Publisher('/masked_depth/depth', Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher('/masked_points', PointCloud2, queue_size=1)

        # Voxel grid map Publish
        self.voxel_pub = rospy.Publisher('/masked_voxel', Int8MultiArray, queue_size=1)

        # voxelization에 필요한 parameter 설정
        self.scene_bound = np.array([[-0.9, -0.9, -0.4],
                                     [0.9, 0.9, 1.4]])  # rough panda workspace
        self.voxel_res = np.array([0.05, 0.05, 0.05]) # [m]
        self.voxel_dim = np.int8(np.ceil((self.scene_bound[1,:] - self.scene_bound[0,:]) / self.voxel_res))

    def camera_info_callback(self, msg):
        # 카메라 매트릭스 정보 추출
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.height, self.width = msg.height, msg.width
        self.fx, self.fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        self.cx, self.cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        self.focal_length = (self.fx+self.fy)/2

    def depth_image_callback(self, msg):
        if self.camera_matrix is None:
            rospy.logwarn("Camera matrix not received yet!")
            return

        try:
            # Depth 이미지를 OpenCV 이미지로 변환
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_image = depth_image.astype(np.float32)

            # 구 모델 정보 추출 및 변환
            all_sphere_centers = []
            all_sphere_radii = []

            for link_name in self.link_names:
                transform = self.get_link_transform("depth_camera_link", link_name)
                if transform:
                    translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
                    orientation = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
                    rot_mat = R.from_quat(orientation).as_matrix()

                    for sphere in self.sphere_models[link_name]:
                        local_center = np.array(sphere['center'])
                        global_center = rot_mat @ local_center + translation
                        all_sphere_centers.append(global_center)
                        all_sphere_radii.append(sphere['radius'])

            all_sphere_centers = np.array(all_sphere_centers)
            all_sphere_radii = np.array(all_sphere_radii)
            
            # 텐서 변환
            all_sphere_centers = torch.tensor(all_sphere_centers, dtype=torch.float32)
            all_sphere_radii = torch.tensor(all_sphere_radii, dtype=torch.float32)

            # Mask 된 depth 이미지와 point clouds 생성
            masked_depth_image, masked_pcl = self.get_masked_depth(all_sphere_radii, all_sphere_centers, torch.tensor(depth_image))

            # OpenCV 이미지를 ROS 이미지 메시지로 변환
            masked_depth_msg = self.bridge.cv2_to_imgmsg(masked_depth_image.numpy(), encoding="passthrough")

            # Masked depth 이미지를 퍼블리시
            self.masked_depth_pub.publish(masked_depth_msg)

            # PointCloud2 메시지 생성
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "depth_camera_link"
            pcl_msg = self.get_pcl_msg(header, masked_pcl.numpy())

            # Point Cloud 퍼블리시
            self.pointcloud_pub.publish(pcl_msg)

            # Point cloud의 frame을 panda frame로 변환
            transform = self.get_link_transform("panda_link0", "depth_camera_link")
            if transform:
                translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
                orientation = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
                rot_mat = R.from_quat(orientation).as_matrix()
                translation = torch.tensor(translation, dtype=torch.float32)
                rot_mat = torch.tensor(rot_mat, dtype=torch.float32)
                transformed_pcl = (rot_mat @ masked_pcl.T).T + translation

                # Point cloud로부터 voxel grid map 생성
                voxel = self.pcl_to_voxel(transformed_pcl)
                voxel_msg = self.get_array_msg(voxel.numpy())
                
                # Voxel array Publish
                self.voxel_pub.publish(voxel_msg)




            
            # import matplotlib.pyplot as plt

            # ax1 = plt.figure(1).add_subplot()
            # ax1.set_title("depth image", fontsize=16, fontweight='bold', pad=20)
            # ax1.imshow(masked_depth_image.numpy())

            # ax2 = plt.figure(2).add_subplot(projection='3d')
            # ax2.voxels(voxel.numpy())
            # axis_res = 0.2 # [m]
            # # ax2.set_xticks(np.arange(0,voxel.shape[0],axis_res/voxel_res))
            # # ax2.set_yticks(np.arange(0,voxel.shape[1],axis_res/voxel_res))
            # # ax2.set_zticks(np.arange(0,voxel.shape[2],axis_res/voxel_res))
            # # ax2.set_xticklabels(np.round(np.arange(scene_bound[0,0],scene_bound[1,0],axis_res),2))
            # # ax2.set_yticklabels(np.round(np.arange(scene_bound[0,1],scene_bound[1,1],axis_res),2))
            # # ax2.set_zticklabels(np.round(np.arange(scene_bound[0,2],scene_bound[1,2],axis_res),2))
            # ax2.set_title("voxel grid", fontsize=16, fontweight='bold', pad=20)
            # ax2.set_xlabel("X")
            # ax2.set_ylabel("Y")
            # ax2.set_zlabel("Z")
            # plt.show()


        except Exception as e:
            rospy.logerr(f"Error in depth_image_callback: {e}")

    def get_pcl_msg(self, header, points):
        # Flatten the points array
        points = points.reshape(-1, 3).astype(np.float32)

        # Define the fields of the PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Create the PointCloud2 message
        pc2_msg = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            fields=fields,
            is_bigendian=False,
            point_step=12,  # 4 bytes per field * 3 fields
            row_step=12 * points.shape[0],
            data=points.tobytes(),
            is_dense=True
        )

        return pc2_msg

    def get_link_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            return transform.transform
        except tf2_ros.LookupException as e:
            rospy.logwarn(f"Could not find the transform from {source_frame} to {target_frame}: {e}")
            return None
        except tf2_ros.ConnectivityException as e:
            rospy.logwarn(f"Connectivity error while looking up transform: {e}")
            return None
        except tf2_ros.ExtrapolationException as e:
            rospy.logwarn(f"Extrapolation error while looking up transform: {e}")
            return None
        
    def get_masked_depth(self, sphere_radii: torch.tensor, sphere_centers: torch.tensor, raw_depth_img: torch.tensor):
        # 깊이 이미지 처리: 구 내부의 값을 inf로 설정
        height, width = list(raw_depth_img.shape)
        u_grid, v_grid = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        u_grid = u_grid.float()
        v_grid = v_grid.float()

        # 카메라 좌표계에서 3D 포인트 계산
        z = raw_depth_img.type(torch.float32)
        x = (u_grid - self.cx) * z / self.fx
        y = (v_grid - self.cy) * z / self.fy
        points_3d = torch.stack((x, y, z), dim=-1).view(-1, 3)

        # 구의 중심과 각 포인트 사이의 거리 계산
        distances = torch.cdist(points_3d, sphere_centers)
        min_distances, _ = distances.min(dim=1)

        # 구 내부의 포인트에 대해 z 값을 inf로 설정
        z_flat = z.view(-1)
        mask = (min_distances < sphere_radii.max()) & (z_flat > 0)
        z_flat[mask] = float('inf')

        # 다시 이미지 형태로 변환
        masked_depth_image = z_flat.view(height, width)

        # Point Cloud 생성
        valid_points = points_3d[z_flat != float('inf')]

        return masked_depth_image, valid_points
    
    def pcl_to_voxel(self, point_cloud):
        """
        Convert point cloud to voxel grid map using PyTorch.

        :param point_cloud: torch tensor of shape (N, 3) where N is the number of points
                            Each row represents a point with (x, y, z) coordinates
        :param min_vals: tuple (min_x, min_y, min_z) representing the minimum values of x, y, z
        :param max_vals: tuple (max_x, max_y, max_z) representing the maximum values of x, y, z
        :param resolution: tuple (res_x, res_y, res_z) representing the voxel grid resolution in each dimension
        :param device: device to use ('cpu' or 'cuda')
        :return: 3D torch tensor representing the voxel grid map
        """

        # Extract min and max values
        min_x, min_y, min_z = self.scene_bound[0,:]
        max_x, max_y, max_z = self.scene_bound[1,:]

        # Extract resolution
        res_x, res_y, res_z = self.voxel_res

        # Calculate voxel grid dimensions
        dim_x = int(np.ceil((max_x - min_x) / res_x))
        dim_y = int(np.ceil((max_y - min_y) / res_y))
        dim_z = int(np.ceil((max_z - min_z) / res_z))

        # Initialize the voxel grid map with zeros
        voxel_grid = torch.zeros(dim_x, dim_y, dim_z, dtype=torch.bool)

        # Convert point cloud coordinates to voxel indices
        voxel_indices = ((point_cloud - torch.tensor([min_x, min_y, min_z])) / 
                        torch.tensor([res_x, res_y, res_z])).floor().long()

        # Ensure voxel indices are within bounds
        voxel_indices = torch.clamp(voxel_indices, min=torch.tensor([0, 0, 0]), max=torch.tensor([dim_x-1, dim_y-1, dim_z-1]))

        # Set the corresponding voxels to 1 (occupied)
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True

        return voxel_grid
    
    def get_array_msg(self, voxel_grid):

        # 메시지 데이터 생성
        array_data = Int8MultiArray()
        
        # 레이아웃 설정
        dim1 = MultiArrayDimension()
        dim1.label = "x"
        dim1.size = voxel_grid.shape[0]
        dim1.stride = voxel_grid.shape[0] * voxel_grid.shape[1] * voxel_grid.shape[2]

        dim2 = MultiArrayDimension()
        dim2.label = "y"
        dim2.size = voxel_grid.shape[1]
        dim2.stride = voxel_grid.shape[1] * voxel_grid.shape[2]

        dim3 = MultiArrayDimension()
        dim3.label = "z"
        dim3.size = voxel_grid.shape[2]
        dim3.stride = voxel_grid.shape[2]

        array_data.layout.dim = [dim1, dim2, dim3]
        array_data.layout.data_offset = 0
        
        # 3D 배열을 1차원 리스트로 변환하여 데이터 채우기
        array_data.data = voxel_grid.flatten().tolist()

        return array_data

        


if __name__ == '__main__':
    processor = DepthImageProcessor()
    rospy.spin()
