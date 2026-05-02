from __future__ import annotations

import numpy as np
import cv2
from typing import Optional


class ObjectDetector:
    """目标检测器：基于颜色的小球检测"""
    
    def __init__(self, color_ranges: dict | None = None) -> None:
        """
        初始化目标检测器
        
        Args:
            color_ranges: 颜色范围字典，格式为:
                {
                    "red": {
                        "lower": [0, 100, 100],
                        "upper": [10, 255, 255]
                    },
                    ...
                }
        """
        if color_ranges is None:
            self.color_ranges = {
                "red": {
                    "lower1": np.array([0, 100, 100]),
                    "upper1": np.array([10, 255, 255]),
                    "lower2": np.array([160, 100, 100]),
                    "upper2": np.array([180, 255, 255])
                },
                "green": {
                    "lower": np.array([35, 100, 100]),
                    "upper": np.array([85, 255, 255])
                },
                "blue": {
                    "lower": np.array([100, 100, 100]),
                    "upper": np.array([140, 255, 255])
                }
            }
        else:
            self.color_ranges = color_ranges
    
    def detect_color_blobs(self, image: np.ndarray, color_name: str = "red",
                           min_area: int = 100) -> tuple[list[tuple[int, int]], np.ndarray]:
        """
        检测指定颜色的色块
        
        Args:
            image: 输入图像 (RGB)
            color_name: 颜色名称
            min_area: 最小面积阈值
            
        Returns:
            centers: 检测到的中心点列表 [(u, v), ...]
            mask: 二值掩码图像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        color_range = self.color_ranges[color_name]
        
        if color_name == "red":
            mask1 = cv2.inRange(hsv, color_range["lower1"], color_range["upper1"])
            mask2 = cv2.inRange(hsv, color_range["lower2"], color_range["upper2"])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
        
        return centers, mask
    
    def detect_ball_position(self, image: np.ndarray, color_name: str = "red") -> tuple[Optional[tuple[float, float]], Optional[float]]:
        """
        检测小球位置（返回最大色块的中心）
        
        Args:
            image: 输入图像 (RGB)
            color_name: 颜色名称
            
        Returns:
            center: 中心点 (u, v)，未检测到返回 None
            radius: 估计半径
        """
        centers, mask = self.detect_color_blobs(image, color_name, min_area=50)
        
        if len(centers) == 0:
            return None, None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            area = cv2.contourArea(max_contour)
            radius = int(np.sqrt(area / np.pi))
            
            return (cx, cy), radius
        
        return None, None
    
    def visualize_detection(self, image: np.ndarray, centers: list[tuple[int, int]],
                            color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 输入图像
            centers: 中心点列表
            color: 绘制颜色
            
        Returns:
            vis_image: 可视化后的图像
        """
        vis_image = image.copy()
        
        for (cx, cy) in centers:
            cv2.circle(vis_image, (cx, cy), 5, color, -1)
            cv2.circle(vis_image, (cx, cy), 20, color, 2)
        
        return vis_image
    
    def filter_by_depth(self, image: np.ndarray, depth_image: np.ndarray,
                        min_depth: float = 0.1, max_depth: float = 2.0) -> np.ndarray:
        """
        根据深度过滤检测结果
        
        Args:
            centers: 中心点列表
            depth_image: 深度图像
            min_depth: 最小深度
            max_depth: 最大深度
            
        Returns:
            filtered_centers: 过滤后的中心点列表
        """
        filtered = []
        
        for (u, v) in centers:
            if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                depth = depth_image[v, u]
                if min_depth <= depth <= max_depth:
                    filtered.append((u, v))
        
        return filtered
