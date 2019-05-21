from __future__ import division

from opencmiss.utils.maths.algorithms import calculate_line_plane_intersection
import cv2

class ImagePlaneModel(object):

    def __init__(self, master_model, video_path):
        self._master_model = master_model
        self._region = None
        self._frames_per_second = -1
        self._images_file_name_listing = []
        self._image_dimensions = [-1, -1]
        self._duration_field = None
        self._image_based_material = None
        self._scaled_coordinate_field = None
        self._time_sequence = []
        self._video_path = video_path

        self._initialise()

    def _initialise(self):
        context = self._master_model.get_context()
        default_region = context.getDefaultRegion()
        self._region = default_region.findChildByName('images')
        field_module = self._region.getFieldmodule()
        self._scaled_coordinate_field = field_module.findFieldByName('scaled_coordinates')
        self._duration_field = field_module.findFieldByName('duration')
        self._image_field = field_module.findFieldByName('volume_image')
        self._image_field = self._image_field.castImage()
        material_module = context.getMaterialmodule()
        self._image_based_material = material_module.findMaterialByName('images')
        self.cap = cv2.VideoCapture(self._video_path)

    def set_image_information(self, image_file_names, frames_per_second, image_dimensions):
        self._images_file_name_listing = image_file_names
        self._frames_per_second = frames_per_second
        self._image_dimensions = image_dimensions

    def get_coordinate_field(self):
        return self._scaled_coordinate_field

    def get_region(self):
        return self._region

    def get_material(self):
        return self._image_based_material

    def get_duration_field(self):
        return self._duration_field

    def get_frame_count(self):
        return len(self._images_file_name_listing)

    def get_frames_per_second(self):
        return self._frames_per_second

    def get_image_file_name_at(self, index):
        return self._images_file_name_listing[index]

    def get_image_at(self, index):
        self.cap.set(1, index)
        res, frame = self.cap.read()
        frame = cv2.flip(frame, 0)
        return frame

    def calculate_image_pixels_rectangle(self, top_left_mesh_location, bottom_right_mesh_location):
        """
        The origin for the rectangle in the image is the top left corner, the mesh locations are given from
        the bottom left corner.
        :param top_left_mesh_location:
        :param bottom_right_mesh_location:
        :return: Rectangle with origin at top left of image described by [x, y, width, height]
        """
        field_module = self._scaled_coordinate_field.getFieldmodule()
        field_module.beginChange()
        field_cache = field_module.createFieldcache()
        field_cache.setMeshLocation(top_left_mesh_location[0], top_left_mesh_location[1])

        _, top_left_values = self._scaled_coordinate_field.evaluateReal(field_cache, 3)
        field_cache.setMeshLocation(bottom_right_mesh_location[0], bottom_right_mesh_location[1])
        _, bottom_right_values = self._scaled_coordinate_field.evaluateReal(field_cache, 3)
        field_module.endChange()

        return (int(top_left_values[0] + 0.5),
                self._image_dimensions[1] - int(top_left_values[1] + 0.5),
                int(bottom_right_values[0] - top_left_values[0] + 0.5),
                int(top_left_values[1] - bottom_right_values[1] + 0.5))

    @staticmethod
    def get_intersection_point(ray):
        return calculate_line_plane_intersection(ray[0], ray[1], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

    def _convert_point_coordinates(self, points):
        return [(point[0], self._image_dimensions[1] - point[1]) for point in points]

    def convert_to_model_coordinates(self, image_points):
        return self._convert_point_coordinates(image_points)

    def convert_to_image_coordinates(self, model_points):
        return self._convert_point_coordinates(model_points)

    def get_time_for_frame_index(self, index):
        frame_count = self.get_frame_count()
        duration = frame_count / self._frames_per_second
        frame_separation = 1 / frame_count
        initial_offset = frame_separation / 2
        return ((index - 1) * frame_separation + initial_offset) * duration

    def get_frame_index_for_time(self, time):
        frame_count = len(self._images_file_name_listing)
        duration = frame_count / self._frames_per_second
        frame_separation = 1 / frame_count
        initial_offset = frame_separation / 2

        frame_id = int((time / duration - initial_offset) / frame_separation + 0.5) + 1

        # self._image_field.setBuffer(self._images_file_name_listing[frame_id])
        self.cap.set(1, frame_id)
        res, frame = self.cap.read()
        frame = cv2.flip(frame, 0)
        self._image_field.setBuffer(frame.tobytes())
        return frame_id

