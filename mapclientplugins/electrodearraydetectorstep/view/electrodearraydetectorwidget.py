from __future__ import division

from PySide import QtGui, QtCore

from opencmiss.zinchandlers.scenemanipulation import SceneManipulation

from mapclientplugins.electrodearraydetectorstep.handlers.datapointadder import DataPointAdder
from mapclientplugins.electrodearraydetectorstep.handlers.datapointremover import DataPointRemover
from mapclientplugins.electrodearraydetectorstep.handlers.rectangletool import RectangleTool
from mapclientplugins.electrodearraydetectorstep.static.strings import DEFINE_ROI_STRING, \
    SET_INITIAL_TRACKING_POINTS_STRING, FINALISE_TRACKING_POINTS_STRING
from mapclientplugins.electrodearraydetectorstep.tools.datapointtool import DataPointTool
from mapclientplugins.electrodearraydetectorstep.tools.trackingtool import TrackingTool
from mapclientplugins.electrodearraydetectorstep.view.ui_electrodearraydetectorwidget \
    import Ui_ElectrodeArrayDetectorWidget

PLAY_TEXT = 'Play'
STOP_TEXT = 'Stop'


class ElectrodeArrayDetectorWidget(QtGui.QWidget):

    def __init__(self, model, parent=None):
        super(ElectrodeArrayDetectorWidget, self).__init__(parent)
        self._ui = Ui_ElectrodeArrayDetectorWidget()
        self._ui.setupUi(model.get_shareable_open_gl_widget(), self)
        self._ui.sceneviewer_widget.set_context(model.get_context())

        self._settings = {'view-parameters': {}}

        self._model = model
        self._model.reset()
        self._model.register_time_value_update_callback(self._update_time_value)
        self._image_plane_scene = model.get_image_plane_scene()
        self._image_plane_scene.create_graphics()
        self._image_plane_scene.set_image_material()
        self._done_callback = None

        self._image_plane_model = model.get_image_plane_model()
        tracking_points_model = model.get_tracking_points_model()
        tracking_points_model.create_model()
        tracking_points_scene = model.get_tracking_points_scene()
        tracking_points_scene.create_graphics()

        self._data_point_tool = DataPointTool(tracking_points_model, self._image_plane_model)
        self._tracking_tool = TrackingTool(model)

        self._setup_handlers()
        self._set_initial_ui_state()
        self._update_ui_state()

        self._load_saved_data = False

        self._make_connections()

    def _make_connections(self):
        self._ui.sceneviewer_widget.graphics_initialized.connect(self._graphics_initialized)
        self._ui.done_pushButton.clicked.connect(self._done_clicked)
        self._ui.timeValue_doubleSpinBox.valueChanged.connect(self._time_value_changed)
        self._ui.timePlayStop_pushButton.clicked.connect(self._time_play_stop_clicked)
        self._ui.timeLoop_checkBox.clicked.connect(self._time_loop_clicked)
        self._ui.detectElectrodes_pushButton.clicked.connect(self._detect_electrodes_button_clicked)
        self._ui.trackElectrodePoints_pushButton.clicked.connect(self._track_electrode_points_button_clicked)
        self._ui.reset_pushButton.clicked.connect(self._reset_button_clicked)

    def _done_clicked(self):
        self._model.done()
        self._done_callback()

    def _graphics_initialized(self):
        """
        Callback for when SceneviewerWidget is initialised
        Set custom scene from model
        """
        scene_viewer = self._ui.sceneviewer_widget.get_zinc_sceneviewer()
        if scene_viewer is not None:
            scene = self._model.get_scene()
            self._ui.sceneviewer_widget.set_tumble_rate(0)
            self._ui.sceneviewer_widget.set_scene(scene)
            if len(self._settings['view-parameters']) == 0:
                self._view_all()
            else:
                eye = self._settings['view-parameters']['eye']
                look_at = self._settings['view-parameters']['look_at']
                up = self._settings['view-parameters']['up']
                angle = self._settings['view-parameters']['angle']
                self._ui.sceneviewer_widget.set_view_parameters(eye, look_at, up, angle)

    def _set_initial_ui_state(self):
        self._ui.timeLoop_checkBox.setChecked(self._model.is_time_loop())
        self._frame_index_value_changed(1)
        self._enter_define_roi()
        minimum_label_width = self._calculate_minimum_label_width()
        self._ui.statusText_label.setMinimumWidth(minimum_label_width)
        maximum_time = self._image_plane_model.get_frame_count() / self._image_plane_model.get_frames_per_second()
        self._ui.timeValue_doubleSpinBox.setMaximum(maximum_time)
        frame_separation = 1 / self._image_plane_model.get_frames_per_second()
        self._ui.timeValue_doubleSpinBox.setDecimals(8)
        self._ui.timeValue_doubleSpinBox.setSingleStep(frame_separation)
        self._ui.timeValue_doubleSpinBox.setValue(frame_separation / 2)

    def _calculate_minimum_label_width(self):
        label = self._ui.statusText_label
        label.setWordWrap(True)
        label.setText(DEFINE_ROI_STRING)
        maximum_width = 0
        width = label.fontMetrics().boundingRect(label.text()).width()
        maximum_width = max(maximum_width, width)
        label.setText(SET_INITIAL_TRACKING_POINTS_STRING)
        width = label.fontMetrics().boundingRect(label.text()).width()
        maximum_width = max(maximum_width, width)
        label.setText(FINALISE_TRACKING_POINTS_STRING)
        width = label.fontMetrics().boundingRect(label.text()).width()
        maximum_width = max(maximum_width, width)
        return maximum_width / 3.0

    def _update_ui_state(self):
        self._ui.statusText_label.setText(DEFINE_ROI_STRING)

    def _setup_handlers(self):
        basic_handler = SceneManipulation()
        self._ui.sceneviewer_widget.register_handler(basic_handler)
        self._rectangle_tool = RectangleTool(QtCore.Qt.Key_D)
        self._data_point_adder = DataPointAdder(QtCore.Qt.Key_A)
        self._data_point_adder.set_model(self._data_point_tool)
        self._data_point_remover = DataPointRemover(QtCore.Qt.Key_D)
        self._data_point_remover.set_model(self._data_point_tool)

    def _reset_button_clicked(self):
        self._tracking_tool.clear()
        self._leave_track_electrode_points()
        self._enter_define_roi()

    def _detect_electrodes_button_clicked(self):
        self._leave_define_roi()
        self._ui.detectElectrodes_pushButton.setEnabled(False)
        self._enter_track_electrode_points()

    def _track_electrode_points_button_clicked(self):
        self._leave_track_electrode_points()
        self._ui.detectElectrodes_pushButton.setEnabled(True)
        self._enter_define_roi()

    def _enter_define_roi(self):
        self._ui.sceneviewer_widget.register_handler(self._rectangle_tool)
        self._ui.sceneviewer_widget.register_key_listener(QtCore.Qt.Key_Return, self._detect_electrodes_button_clicked)
        self._ui.detectElectrodes_pushButton.setEnabled(True)
        self._ui.trackElectrodePoints_pushButton.setEnabled(False)
        self._ui.reset_pushButton.setEnabled(False)

    def _leave_define_roi(self):
        rectangle_description = self._rectangle_tool.get_rectangle_box_description()
        if sum(rectangle_description) < 0:
            QtGui.QMessageBox.warning(self, 'Invalid ROI', 'The region of interest is invalid and region'
                                      ' analysis will not be performed')
        else:
            self._rectangle_tool.remove_rectangle_box()
            self._ui.sceneviewer_widget.unregister_handler(self._rectangle_tool)
            self._ui.sceneviewer_widget.unregister_key_listener(QtCore.Qt.Key_Return)

            x = rectangle_description[0]
            y = rectangle_description[1]
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            element = self._ui.sceneviewer_widget.get_nearest_element(x, y)
            QtGui.QApplication.restoreOverrideCursor()
            if element.isValid():
                QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                image_index = self._model.get_frame_index()
                if self._load_saved_data:
                    self._tracking_tool.load_saved_data()
                else:
                    self._tracking_tool.analyse_roi(
                        image_index, self._ui.sceneviewer_widget.get_zinc_sceneviewer(), element, rectangle_description)
                QtGui.QApplication.restoreOverrideCursor()
            else:
                QtGui.QMessageBox.warning(self, 'Invalid ROI', 'The region of interest is invalid and region'
                                          ' analysis will not be performed')

    def _enter_track_electrode_points(self):
        self._ui.sceneviewer_widget.register_handler(self._data_point_adder)
        self._ui.sceneviewer_widget.register_handler(self._data_point_remover)
        self._ui.sceneviewer_widget.register_key_listener(QtCore.Qt.Key_Return,
                                                          self._track_electrode_points_button_clicked)
        self._ui.detectElectrodes_pushButton.setEnabled(False)
        self._ui.trackElectrodePoints_pushButton.setEnabled(True)
        self._ui.reset_pushButton.setEnabled(True)

    def _leave_track_electrode_points(self):
        self._ui.sceneviewer_widget.unregister_handler(self._data_point_adder)
        self._ui.sceneviewer_widget.unregister_handler(self._data_point_remover)
        self._ui.sceneviewer_widget.unregister_key_listener(QtCore.Qt.Key_Return)

        # Perform the tracking for all images.
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self._tracking_tool.track_key_points()
        QtGui.QApplication.restoreOverrideCursor()

    def _enter_finalise_tracking_points(self):
        self._ui.sceneviewer_widget.register_handler(self._data_point_adder)
        self._ui.sceneviewer_widget.register_handler(self._data_point_remover)

    def _leave_finalise_tracking_points(self):
        self._ui.sceneviewer_widget.unregister_handler(self._data_point_adder)
        self._ui.sceneviewer_widget.unregister_handler(self._data_point_remover)

    def _view_all(self):
        if self._ui.sceneviewer_widget.get_zinc_sceneviewer() is not None:
            self._ui.sceneviewer_widget.view_all()

    def register_done_callback(self, done_callback):
        self._done_callback = done_callback

    def set_settings(self, settings):
        self._settings.update(settings)

    def get_settings(self):
        eye, look_at, up, angle = self._ui.sceneviewer_widget.get_view_parameters()
        self._settings['view-parameters'] = {'eye': eye, 'look_at': look_at, 'up': up, 'angle': angle}
        return self._settings

    def set_images_info(self, images_info):
        self._image_plane_model.load_images(images_info)
        self._image_plane_scene.set_image_material()
        frame_count = self._image_plane_model.get_frame_count()
        value = self._model.get_frames_per_second()
        duration = frame_count / value
        self._ui.timeValue_doubleSpinBox.setMaximum(duration)
        self._model.set_maximum_time_value(duration)
        self._model.set_frame_index(2)

    def _update_time_value(self, value):
        self._ui.timeValue_doubleSpinBox.blockSignals(True)
        frame_count = self._image_plane_model.get_frame_count()
        max_time_value = frame_count / self._image_plane_model.get_frames_per_second()

        if value > max_time_value:
            self._ui.timeValue_doubleSpinBox.setValue(max_time_value)
            self._time_play_stop_clicked()
        else:
            self._ui.timeValue_doubleSpinBox.setValue(value)
        self._ui.timeValue_doubleSpinBox.blockSignals(False)

    def _time_value_changed(self, value):
        self._model.set_time_value(value)

    def _time_play_stop_clicked(self):
        current_text = self._ui.timePlayStop_pushButton.text()
        if current_text == PLAY_TEXT:
            self._ui.timePlayStop_pushButton.setText(STOP_TEXT)
            self._model.play()
        else:
            self._ui.timePlayStop_pushButton.setText(PLAY_TEXT)
            self._model.stop()

    def _time_loop_clicked(self):
        self._model.set_time_loop(self._ui.timeLoop_checkBox.isChecked())

    def _frame_index_value_changed(self, value):
        self._model.set_frame_index(value)
