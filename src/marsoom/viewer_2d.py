from typing import Optional, Tuple, Union
import enum

import numpy as np
import torch
from imgui_bundle import imgui, ImVec2, ImVec4

from marsoom.texture import Texture


class eViewerUnit(enum.Enum):
	WINDOW = 1
	PIXELS = 2
	UNIT = 3
	CANVAS = 4

class Viewer2D:
	def __init__(
			self, 
			name:str, 
			allow_pan: bool = True, 
			allow_zoom: bool = True, 
			pixels_to_units: np.ndarray = np.eye(3, dtype=np.float32),
			size: Optional[Tuple[int, int]] = None):
		self.name = name
		self.image_texture = None
		self.pixels_to_uv = np.eye(3)
		self.canvas_to_pixels = np.eye(3)
		self.size = size
		self.canvas_location_in_window_space = np.array([0, 0])
		self.window_to_canvas = np.eye(3)
		self.pixels_to_uv = np.eye(3)
		self.hovered = False
		self.mouse_pos_pixels = np.array([0, 0])
		self.mouse_delta = np.array([0, 0])
		self.allow_pan = allow_pan
		self.allow_zoom = allow_zoom
		self.moving = []
		self.pixels_to_units = pixels_to_units
		self.units_to_pixels = np.linalg.inv(self.pixels_to_units)

	@property
	def pixels_to_canvas(self):
		return np.linalg.inv(self.canvas_to_pixels)
	
	@property
	def window_to_pixels(self):
		return self.canvas_to_pixels @ self.window_to_canvas

	@property
	def pixels_to_window(self):
		return np.linalg.inv(self.window_to_pixels)
	
	@property
	def units_to_window(self):
		return self.pixels_to_window @ self.units_to_pixels
	
	@property
	def window_to_units(self):
		return self.pixels_to_units @ self.window_to_pixels 

	@property
	def canvas_to_window(self):
		return self.pixels_to_window @ self.canvas_to_pixels
	
	def update_image(self, image: torch.Tensor):
		"""
		image: torch.Tensor of shape [H, W, 3] with dtype float32 and range between 0 and 1
		"""
		if self.image_texture is None:
			self.image_texture = Texture(image.shape[1], image.shape[0])
			self.pixels_to_uv[0, 0] = 1.0 / image.shape[1]
			self.pixels_to_uv[1, 1] = 1.0 / image.shape[0]

		self.image_texture.copy_from_device(image)

	def _draw_image(self):
		window_location = imgui.get_window_pos()
		self.window_to_canvas[0, 2] = -window_location.x
		self.window_to_canvas[1, 2] = -window_location.y
		tl = imgui.get_cursor_screen_pos()
		if self.size:
			s = self.size
		else:
			s = imgui.get_content_region_avail()

		# self.canvas_location_in_window_space = np.array((tl.x, tl.y))
		self.canvas_location_in_window_space = np.array((0.0, 0.0))
		tl = np.array((tl.x, tl.y, 1))
		br = np.array((tl[0] + s[0], tl[1] + s[1], 1))
		tl_uv = self.pixels_to_uv @ self.window_to_pixels @ tl
		br_uv = self.pixels_to_uv @ self.window_to_pixels @ br
		tl_uv = ImVec2(tl_uv[0], tl_uv[1])
		br_uv = ImVec2(br_uv[0], br_uv[1])
		id = self.image_texture.id if self.image_texture is not None else 0
		imgui.image(
				id,
				s,
				tl_uv,
				br_uv
				)
		self.hovered = imgui.is_item_hovered()

	def process_mouse(self):
		io = imgui.get_io()
		# middle mouse button
		if not io.mouse_down[0]:
			self.moving = []
		if self.hovered:
			# mouse_pos = np.array([io.mouse_pos.x, io.mouse_pos.y])
			mouse_pos = self.window_to_canvas @ np.array([io.mouse_pos.x, io.mouse_pos.y, 1])
			mouse_pos = mouse_pos[:2]
			if self.allow_zoom and io.mouse_wheel:
				zoom_ratio = np.exp(io.mouse_wheel/4.0)
				self.canvas_to_pixels = self.canvas_to_pixels @ zoom_matrix(mouse_pos, zoom_ratio)
			if self.allow_pan and io.mouse_down[2] and io.mouse_delta:
				mouse_drag = np.array([-io.mouse_delta[0], -io.mouse_delta[1]])
				self.canvas_to_pixels = self.canvas_to_pixels @ pan_matrix(mouse_drag, 1.0)

			self.mouse_delta = self.canvas_to_pixels[:2, :2] @ np.array([io.mouse_delta[0], io.mouse_delta[1]])
			self.mouse_pos_pixels = self.canvas_to_pixels[:2, :2] @ mouse_pos + self.canvas_to_pixels[:2, 2]
		
	def get_mouse_position(self, unit: eViewerUnit = eViewerUnit.PIXELS):
		io = imgui.get_io()
		window_to_unit = None
		if unit == eViewerUnit.PIXELS:
			window_to_unit = self.window_to_pixels
		elif unit == eViewerUnit.UNIT:
			window_to_unit = self.window_to_units
		elif unit == eViewerUnit.WINDOW:
			window_to_unit = np.eye(3, np.float32)
		elif unit == eViewerUnit.CANVAS:
			window_to_unit = self.window_to_canvas
		else:
			raise ValueError("Unknown unit")
		res = window_to_unit @ np.array([io.mouse_pos.x, io.mouse_pos.y, 1])
		return res.flatten()[:2]


	def _get_unit_to_window(self, unit: eViewerUnit):
		base_to_window = None
		if unit == eViewerUnit.PIXELS:
			base_to_window = self.pixels_to_window
		elif unit == eViewerUnit.UNIT:
			base_to_window = self.units_to_window
		elif unit == eViewerUnit.CANVAS:
			base_to_window = self.canvas_to_window
		elif unit == eViewerUnit.WINDOW:
			base_to_window = np.eye(3, np.float32)
		else:
			raise ValueError("Unknown unit")
		return base_to_window

	def _convert_to_window_position(self, points: np.ndarray, unit: eViewerUnit):
		base_to_window = self._get_unit_to_window(unit)
		points = np.asarray(points).reshape(-1, 2)
		return (base_to_window[:2, :2] @ points.T + base_to_window[:2, 2].reshape(2,1)).T

	def _convert_to_window_scale(self, scale: np.ndarray, unit: eViewerUnit):
		base_to_window = self._get_unit_to_window(unit)
		points = np.asarray(scale).reshape(-1, 1).repeat(2, axis=-1)
		return (base_to_window[:2, :2] @ points.T).T
	
	def line(self,
			 positions: Tuple[Tuple[float, float], Tuple[float, float]],
			 color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0), 
			 thickness=2.0, 
			 unit: eViewerUnit = eViewerUnit.PIXELS
			 ):
		points_canvas = self._convert_to_window_position(positions, unit=unit)
		p1 = ImVec2(points_canvas[0, 0], points_canvas[0, 1])
		p2 = ImVec2(points_canvas[1, 0], points_canvas[1, 1])
		imgui.get_window_draw_list().add_line(
			p1, 
			p2,
			imgui.color_convert_float4_to_u32(ImVec4(*color)),
			thickness=thickness,
			)

	def polyline(self,
			 positions: list[Tuple[float, float]],
			 color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0), 
			 thickness=2.0, 
			 hover_threshold=30.0, 
			 hover_color=(0.0, 1.0, 0.0, 1.0),
			 unit: eViewerUnit = eViewerUnit.PIXELS
			 ):
		points_canvas = self._convert_to_window_position(positions, unit=unit)
		points = [ImVec2(point[0], point[1]) for point in points_canvas]
		# check if mouse is near center
		center = np.mean(points_canvas, axis=0)
		hovered = imgui.is_mouse_hovering_rect(ImVec2(center[0] - hover_threshold, center[1] - hover_threshold), ImVec2(center[0] + hover_threshold, center[1] + hover_threshold))
		# points = points + [ImVec2(points_canvas[0][0], points_canvas[0][1])]
		color = color
		if hovered:
			color = hover_color
		imgui.get_window_draw_list().add_polyline(
			points,
			imgui.color_convert_float4_to_u32(ImVec4(*color)),
			thickness=thickness,
			flags=0,
			)
		return hovered
	
	def is_moving(self, name: str):
		return name in self.moving
	
	def can_move(self, name: str):
		if len(self.moving) == 0:
			return True
		if name in self.moving:
			return True
		return False
	
	def add_to_moving(self, name: str):
		if name not in self.moving:
			self.moving.append(name)

	# def control_quad(
	#         self, 
	#         name: str,
	#         points_px: Union[list[list[float]], np.ndarray], 
	#         color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0), 
	#         thickness=2.0,
	#         control_threshold=5.0 # if within 5 pixels of a control point, allow user to adjust
	#         ):
	#     changed = False
	#     points_px = np.array(points_px, dtype=np.float32)


	#     for i, point in enumerate(points_px):
	#         cname = f"{name}_control_point_{i}"
	#         self.text(f"{i}", point, color)
	#         hovered = self.circle(cname, point, color, hover_threshold=10)
	#         if (hovered or self.is_moving(cname)) and imgui.is_mouse_down(0) and self.can_move(cname):
	#             self.add_to_moving(cname)
	#             points_px[i] += self.mouse_delta
	#             changed = True

	#     quad_name = f"{name}_quad"
	#     hovered = self.quad(quad_name, points_px, color, thickness)
	#     if (hovered or self.is_moving(quad_name)) and imgui.is_mouse_down(0) and self.can_move(quad_name):
	#         self.add_to_moving(quad_name)
	#         points_px += self.mouse_delta
	#         changed = True
	#         # move quad by dx

	#     return changed, points_px

	def circle(self, 
			   position: Tuple[float, float], 
			   color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0), 
			   hover_threshold=10.0, 
			   radius=5.0, 
			   thickness=1.0,
			   unit: eViewerUnit = eViewerUnit.PIXELS):

		point_canvas = self._convert_to_window_position(position, unit=unit)
		radius_canvas = self._convert_to_window_scale(radius, unit=unit)[0, 0]
		# thickness_canvas = self._convert_to_window_scale(thickness, unit=unit)[0, 0]
		thickness_canvas = thickness

		hovered = imgui.is_mouse_hovering_rect(ImVec2(point_canvas[0, 0] - hover_threshold, point_canvas[0, 1] - hover_threshold), ImVec2(point_canvas[0, 0] + hover_threshold, point_canvas[0, 1] + hover_threshold))
		if hovered: 
			color = (1.0, 0.0, 0.0, 1.0)

		imgui.get_window_draw_list().add_circle(
			ImVec2(point_canvas[0, 0], point_canvas[0, 1]), radius_canvas, imgui.color_convert_float4_to_u32(ImVec4(*color)), thickness=thickness_canvas
		)
		return hovered
	
	def text(self, 
			 text: str, 
			 position: Tuple[float, float], 
			 color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0), 
			 font_scale=1.0, 
			 unit: eViewerUnit = eViewerUnit.PIXELS):
		point_canvas = self._convert_to_window_position(position, unit=unit)
		imgui.set_window_font_scale(font_scale)
		imgui.get_window_draw_list().add_text(
			ImVec2(point_canvas[0, 0], point_canvas[0, 1]), imgui.color_convert_float4_to_u32(ImVec4(*color)), text
		)
		imgui.set_window_font_scale(1.0)
	

	def axis(self, scale = 1.0, unit = eViewerUnit.PIXELS, thickness=3.0):
		self.line(
			((0.0, 0.0), (scale, 0.0)),
			thickness=thickness,
			color=(1, 0, 0, 1),
			unit=unit
		)
		self.line(
			((0.0, 0.0), (0.0, scale)),
			thickness=thickness,
			color=(0, 1, 0, 1),
			unit=unit
		)
	
	def draw(self):
		# if self.image_texture is not None:
		self.process_mouse()
		self._draw_image()

def zoom_matrix(center: np.ndarray, ratio: float):
	#https://github.com/pthom/immvision/blob/b9791af4b1a42326db89e9f1305013f554a0f450/src/immvision/internal/cv/zoom_pan_transform.cpp#L17
	zoom = np.eye(3)
	zoom[0, 0] = ratio
	zoom[1, 1] = ratio
	zoom[0, 2] = (1.0 - ratio) * center[0]
	zoom[1, 2] = (1.0 - ratio) * center[1]
	return zoom

def pan_matrix(drag_delta: np.ndarray, current_zoom: float):
	pan = np.eye(3)
	pan[0, 2] = drag_delta[0] / current_zoom
	pan[1, 2] = drag_delta[1] / current_zoom
	return pan 

def compute_affine_transform(p0: Tuple[float, float], p0_map: Tuple[float, float], p1: Tuple[float, float], p1_map: Tuple[float, float], p2: Tuple[float, float], p2_map: Tuple[float, float], flip = False):
	# Create the source and destination points as numpy arrays
	src = np.array([p0, p1, p2]) # pixels
	dst = np.array([p0_map, p1_map, p2_map]) # meters
	
	# Construct the transformation matrix using the two points
	A = np.array([
		[src[0][0], src[0][1], 1, 0, 0, 0],
		[0, 0, 0, src[0][0], src[0][1], 1],
		[src[1][0], src[1][1], 1, 0, 0, 0],
		[0, 0, 0, src[1][0], src[1][1], 1],
		[src[2][0], src[2][1], 1, 0, 0, 0],
		[0, 0, 0, src[2][0], src[2][1], 1],

	])
	
	B = np.array(dst).flatten()
	
	# Solve for the transformation coefficients
	# print(A, B)
	coeffs = np.linalg.solve(A, B)
	
	# The coefficients correspond to the transformation matrix
	transform_matrix = np.array([
		[coeffs[0], coeffs[1], coeffs[2]],
		[coeffs[3], coeffs[4], coeffs[5]],
		[0, 0, 1]
	])
	if flip:
		transform_matrix = np.array([
			[0, 1, 0],
			[1, 0, 0],
			[0, 0, 1]
		]) @ transform_matrix

	return transform_matrix