import numpy as np
import mujoco


class Camera:
    """MuJoCo 相机类：获取仿真环境中的图像"""
    
    def __init__(self, model, data, camera_name="camera", width=640, height=480):
        """
        初始化相机
        
        Args:
            model: MuJoCo 模型
            data: MuJoCo 数据
            camera_name: 相机名称
            width: 图像宽度
            height: 图像高度
        """
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height
        
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        
        self.renderer = mujoco.Renderer(model, height, width)
    
    def get_image(self):
        """
        获取当前帧的 RGB 图像
        
        Returns:
            image: RGB 图像数组 [height, width, 3]
        """
        self.renderer.update_scene(self.data, self.camera_id)
        image = self.renderer.render()
        return image
    
    def get_depth(self):
        """
        获取深度图像
        
        Returns:
            depth: 深度图像数组 [height, width]
        """
        self.renderer.update_scene(self.data, self.camera_id)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth
    
    def get_camera_params(self):
        """
        获取相机内参
        
        Returns:
            fx, fy: 焦距
            cx, cy: 主点坐标
        """
        cam = self.model.cam(self.camera_id)
        fovy = cam.fovy[0]
        
        fy = self.height / (2 * np.tan(np.radians(fovy) / 2))
        fx = fy
        
        cx = self.width / 2
        cy = self.height / 2
        
        return fx, fy, cx, cy
    
    def get_camera_pose(self):
        """
        获取相机在世界坐标系中的位姿
        
        Returns:
            position: 相机位置 [x, y, z]
            orientation: 相机旋转矩阵 [3x3]
        """
        position = self.data.cam_xpos[self.camera_id].copy()
        orientation = self.data.cam_xmat[self.camera_id].copy().reshape(3, 3)
        return position, orientation
    
    def pixel_to_ray(self, u, v):
        """
        将像素坐标转换为相机坐标系中的射线
        
        Args:
            u, v: 像素坐标
            
        Returns:
            ray: 射线方向向量
        """
        fx, fy, cx, cy = self.get_camera_params()
        
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = 1.0
        
        ray = np.array([x, y, z])
        ray = ray / np.linalg.norm(ray)
        
        return ray
    
    def pixel_to_world(self, u, v, depth):
        """
        将像素坐标和深度转换为世界坐标
        
        Args:
            u, v: 像素坐标
            depth: 深度值
            
        Returns:
            world_pos: 世界坐标 [x, y, z]
        """
        ray = self.pixel_to_ray(u, v)
        
        cam_pos, cam_rot = self.get_camera_pose()
        
        world_ray = cam_rot @ ray
        world_pos = cam_pos + world_ray * depth
        
        return world_pos
