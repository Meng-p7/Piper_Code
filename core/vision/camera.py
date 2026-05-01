import numpy as np
import mujoco


class Camera:
    def __init__(self, model, data, camera_name="camera", width=640, height=480):
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height
        
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.renderer = mujoco.Renderer(model, height, width)
    
    def get_image(self):
        self.renderer.update_scene(self.data, self.camera_id)
        image = self.renderer.render()
        return image
    
    def get_depth(self):
        self.renderer.update_scene(self.data, self.camera_id)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth
    
    def get_camera_params(self):
        cam = self.model.cam(self.camera_id)
        fovy = cam.fovy[0]
        
        fy = self.height / (2 * np.tan(np.radians(fovy) / 2))
        fx = fy
        
        cx = self.width / 2
        cy = self.height / 2
        
        return fx, fy, cx, cy
    
    def get_camera_pose(self):
        position = self.data.cam_xpos[self.camera_id].copy()
        orientation = self.data.cam_xmat[self.camera_id].copy().reshape(3, 3)
        return position, orientation
    
    def pixel_to_world(self, u, v, depth):
        """
        将像素坐标和深度转换为世界坐标
        
        Args:
            u, v: 像素坐标
            depth: 深度值（相机 Z 轴距离）
            
        Returns:
            world_pos: 世界坐标 [x, y, z]
        """
        fx, fy, cx, cy = self.get_camera_params()
        
        x_cam = (u - cx) / fx * depth
        y_cam = (v - cy) / fy * depth
        z_cam = depth
        
        cam_pos, cam_rot = self.get_camera_pose()
        
        cam_point = np.array([x_cam, y_cam, z_cam])
        world_pos = cam_pos + cam_rot @ cam_point
        
        return world_pos
