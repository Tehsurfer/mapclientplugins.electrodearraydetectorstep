from __future__ import division

from PySideX import QtCore
from opencmiss.utils.zinc import defineStandardVisualisationTools

from mapclientplugins.electrodearraydetectorstep.model.imageplanemodel import ImagePlaneModel
from mapclientplugins.electrodearraydetectorstep.model.trackingpointsmodel import TrackingPointsModel
from mapclientplugins.electrodearraydetectorstep.scene.imageplanescene import ImagePlaneScene
from mapclientplugins.electrodearraydetectorstep.scene.trackingpointsscene import TrackingPointsScene


class MasterModel(object):

    def __init__(self, image_context_data):
        self._settings = {
            'time-loop': False
        }

        self._context = image_context_data.get_context()
        self._shareable_widget = image_context_data.get_shareable_open_gl_widget()
        self._image_buffer = image_context_data.get_image_file_names()
        defineStandardVisualisationTools(self._context)
        self._default_region = self._context.getDefaultRegion()

        timekeeper_module = self._context.getTimekeepermodule()
        self._timekeeper = timekeeper_module.getDefaultTimekeeper()
        self._timer = QtCore.QTimer()
        self._current_time = 0.0
        self._time_value_update = None
        self._frame_index_update = None

        self._image_plane_model = ImagePlaneModel(self, image_context_data.get_video_path())
        self._image_plane_model.set_image_information(image_context_data.get_image_file_names(),
                                                      image_context_data.get_frames_per_second(),
                                                      image_context_data.get_image_dimensions())
        self._tracking_points_model = TrackingPointsModel(self)
        self._image_plane_scene = ImagePlaneScene(self)
        self._tracking_points_scene = TrackingPointsScene(self)

        self.set_maximum_time_value(image_context_data.get_frame_count() / image_context_data.get_frames_per_second())

        self._make_connections()

    def _make_connections(self):
        self._timer.timeout.connect(self._timeout)

    def _timeout(self):
        increment = 1000 / self._image_plane_model.get_frames_per_second() / 1000
        duration = self._image_plane_model.get_frame_count() / self._image_plane_model.get_frames_per_second()
        if not self._settings['time-loop'] and (self._current_time + increment) > duration:
            self._current_time = duration + 1e-08
        else:
            self._current_time += increment
        if self._settings['time-loop'] and self._current_time > duration:
            self._current_time -= duration

        self._timekeeper.setTime(self._current_time)
        self._time_value_update(self._current_time)
        # frame_index = self._image_plane_model.get_frame_index_for_time(self._current_time) + 1
        # self.set_frame_index(frame_index)

    def register_frame_index_update_callback(self, frame_index_update_callback):
        self._frame_index_update = frame_index_update_callback

    def register_time_value_update_callback(self, time_value_update_callback):
        self._time_value_update = time_value_update_callback

    def set_frame_index(self, frame_index):
        self._current_time = self._image_plane_model.get_time_for_frame_index(frame_index)
        self._timekeeper.setTime(self._current_time)
        self._time_value_update(self._current_time)

    def get_frame_index(self):
        return self._image_plane_model.get_frame_index_for_time(self._current_time)

    def set_time_value(self, time):
        self._current_time = time
        self._timekeeper.setTime(time)
        # frame_index = self._image_plane_model.get_frame_index_for_time(time) + 1
        # self._frame_index_update(frame_index)

    def get_time_sequence(self):
        time_sequence = []
        for frame_value in range(self._image_plane_model.get_frame_count()):
            time = self._image_plane_model.get_time_for_frame_index(frame_value + 1)
            time_sequence.append(time)

        return time_sequence

    def set_time_loop(self, state):
        self._settings['time-loop'] = state

    def is_time_loop(self):
        return self._settings['time-loop']

    def play(self):
        self._timer.start(1000 / self._image_plane_model.get_frames_per_second())

    def stop(self):
        self._timer.stop()

    def done(self):
        pass

    def get_context(self):
        return self._context

    def get_shareable_open_gl_widget(self):
        return self._shareable_widget

    def get_default_region(self):
        return self._default_region

    def get_scene(self):
        return self._default_region.getScene()

    def get_timekeeper(self):
        return self._timekeeper

    def get_timekeeper_time(self):
        return self._timekeeper.getTime()

    def set_maximum_time_value(self, maximum_time_value):
        self._timekeeper.setMaximumTime(maximum_time_value)

    def set_time(self, value):
        pass

    def get_image_plane_model(self):
        return self._image_plane_model

    def get_tracking_points_model(self):
        return self._tracking_points_model

    def get_image_plane_scene(self):
        return self._image_plane_scene

    def get_tracking_points_scene(self):
        return self._tracking_points_scene

    def reset(self):
        self._tracking_points_model.create_model()
        self._tracking_points_scene.create_graphics()

    def set_settings(self, settings):
        self._settings.update(settings)

    def get_settings(self):
        return self._settings

    def get_image_buffer(self):
        return self._image_buffer
