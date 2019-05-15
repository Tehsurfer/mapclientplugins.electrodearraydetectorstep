
from bisect import bisect_left

from opencmiss.zinc.status import OK as CMISS_OK
from opencmiss.utils.zinc import create_finite_element_field, create_node, AbstractNodeDataObject
import numpy as np

class NodeCreator(AbstractNodeDataObject):

    def __init__(self, coordinates, index=''):
        super(NodeCreator, self).__init__(['coordinates'])
        self._coordinates = coordinates
        self._index = index

    def coordinates(self):
        return self._coordinates

    def index(self):
        return self._index

    def set_index(self, index):
        self._index = index


class KeyPoint(object):

    def __init__(self, node, time):
        self._node = node
        self._creation_time = time
        self._label = '%s' % self._node.getIdentifier()

    def get_creation_time(self):
        return self._creation_time

    def set_label(self, label):
        self._label = label

    def get_node(self):
        return self._node

    def has_node(self, node):
        return node.getIdentifier() == self._node.getIdentifier()

    def get_label(self):
        return self._label


class ElectrodeKeyPoint(KeyPoint):
    pass


class SegmentedKeyPoint(KeyPoint):
    pass


class TrackingPointsModel(object):

    def __init__(self, master_model):
        self._master_model = master_model
        self._region = None
        self._coordinate_field = None
        self._index_field = None
        self._selection_group = None
        self._selection_group_field = None
        self._logger = master_model.get_context().getLogger()
        self._key_points = []

    def get_region(self):
        return self._region

    def get_coordinate_field(self):
        return self._coordinate_field

    def get_index_field(self):
        return self._index_field

    def select_node(self, identifier):
        node = self._get_node(identifier)
        self._selection_group.removeAllNodes()
        self._selection_group.addNode(node)

    def deselect_node(self, identifier):
        node = self._get_node(identifier)
        self._selection_group.removeNode(node)

    def is_selected(self, identifier):
        node = self._get_node(identifier)
        return self._selection_group.containsNode(node)

    def _create_node(self, location, time, index=None):
        field_module = self._coordinate_field.getFieldmodule()
        node_creator = NodeCreator(location)
        node_creator.set_time_sequence(self._master_model.get_time_sequence())
        node_creator.set_time_sequence_field_names(['coordinates'])
        if index is not None:
            node_creator.set_field_names(['coordinates', 'index'])
            node_creator.set_index(index)
            node_creator.set_time_sequence_field_names(['coordinates'])
        identifier = create_node(field_module, node_creator,
                                 node_set_name='datapoints', time=time)

        return self._get_node(identifier)

    def set_node_location(self, node, location):
        time = self._master_model.get_timekeeper_time()
        field_module = self._coordinate_field.getFieldmodule()
        field_module.beginChange()
        field_cache = field_module.createFieldcache()
        field_cache.setTime(time)
        field_cache.setNode(node)
        self._coordinate_field.assignReal(field_cache, location)
        field_module.endChange()

    def get_key_points_description(self):
        description = {}

        time_array = self._master_model.get_time_sequence()
        description['time_array'] = time_array

        field_module = self._coordinate_field.getFieldmodule()
        field_module.beginChange()
        field_cache = field_module.createFieldcache()
        for key_point in self._key_points:
            node = key_point.get_node()
            field_cache.setNode(node)
            node_locations = []
            for time in time_array:
                field_cache.setTime(time)
                _, coordinates = self._coordinate_field.evaluateReal(field_cache, 3)

                node_locations.append(coordinates)

            description[key_point.get_label()] = node_locations

        ordered_description = self._order_node_numbers(description)
        return ordered_description

    def _order_node_numbers(self, description):

        # Format out of dictionary
        node_array = []
        for i in range(1, 65):
            coords = description[f'{i}'][0]
            coords.append(i)
            node_array.append(coords)

        # Sort by X and Y values
        sortx = sorted(node_array, key=lambda node_array: node_array[0])
        sorty = sorted(node_array, key=lambda node_array: node_array[1])

        # Add to sorted list based off max(y) while progressively excluding min(x) values
        sorted_list = []
        storage_list = []
        j = 0

        # Our loop ends if all elements have been alocated
        while len(sortx) != 0:

            # Figure what our minimum y value is
            if j == 0:
                miny = 0
            elif (j * 8 - len(storage_list) - len(sorted_list) - 1) <= 0:
                miny = 0
            else:
                miny = j * 8 - len(storage_list) - len(sorted_list) - 1

            # Check if we can add this point to the ordered list next
            add_next = False
            if sortx[0] in sorty[miny:]:
                if j == 0 or sorted_list[-1][1] < sortx[0][1]:
                    add_next = True

            # Add it to sorted list if we can
            if add_next:
                sorted_list.append(sortx[0])
                sorty.remove(sortx[0])
                sortx.remove(sortx[0])
                j = j + 1
            # otherwise leave it out of the search until we get to the next row
            else:
                storage_list.append(sortx[0])
                sorty.remove(sortx[0])
                sortx.remove(sortx[0])

            # Reset our search at the end of a row
            if j > 8:
                if len(storage_list) > 0 and len(sortx) > 0:
                    sortx = np.concatenate((sortx, storage_list)).tolist()
                    sorty = np.concatenate((sorty, storage_list)).tolist()
                    storage_list = []
                sortx = sorted(sortx, key=lambda sortx: sortx[0])
                sorty = sorted(sorty, key=lambda sorty: sorty[1])
                j = 0

            # Check for the case where our search has gone awry
            if len(sortx) == 0 and len(storage_list) >= 8:
                sortx = storage_list
                sorty = storage_list
                storage_list = []
                sortx = sorted(sortx, key=lambda sortx: sortx[0])
                sorty = sorted(sorty, key=lambda sorty: sorty[1])
                j = 0

        # Add remaining values if search went awry
        if len(storage_list) > 0:
            sorted_list = np.concatenate((sorted_list, storage_list)).tolist()

        # Modify our dictionary
        ordered_description = {}
        for i, new_index in enumerate(sorted_list):
            ordered_description[f'{i + 1}'] = description[f'{int(new_index[3])}']
        ordered_description['time_array'] = description['time_array']

        return ordered_description

    def remove_node(self, identifier):
        node = self._get_node(identifier)
        key_points = [point for point in self._key_points if point.has_node(node)]
        key_point_index = self._key_points.index(key_points[0])
        self._key_points.pop(key_point_index)
        node_set = node.getNodeset()
        node_set.destroyNode(node)

    def _get_node(self, identifier):
        node_set = self._selection_group.getMasterNodeset()
        return node_set.findNodeByIdentifier(identifier)

    def get_selection_field(self):
        return self._selection_group_field

    def create_segmented_key_point(self, location):
        time = self._master_model.get_timekeeper_time()
        node = self._create_node(location, time)
        self.select_node(node.getIdentifier())
        self._key_points.append(SegmentedKeyPoint(node, time))

    def create_electrode_key_points(self, key_points):
        time = self._master_model.get_timekeeper_time()
        node_time = _get_nearest_match(self._master_model.get_time_sequence(), time)
        field_module = self._coordinate_field.getFieldmodule()
        field_module.beginChange()
        for index, key_point in enumerate(key_points):
            x = float(key_point[0])
            y = float(key_point[1])
            node = self._create_node([x, y, 0.0], node_time, index='{0}'.format(index))
            self._key_points.append(ElectrodeKeyPoint(node, node_time))
        field_module.endChange()

    def set_key_points_at_time(self, key_points, time):
        assert len(key_points) == len(self._key_points)
        field_module = self._coordinate_field.getFieldmodule()
        field_module.beginChange()
        field_cache = field_module.createFieldcache()
        field_cache.setTime(time)
        for index, key_point in enumerate(self._key_points):
            node = key_point.get_node()
            coordinates = [key_points[index][0], key_points[index][1], 0.0]
            field_cache.setNode(node)
            self._coordinate_field.assignReal(field_cache, coordinates)

        field_module.endChange()

    def get_key_points(self):
        key_points = []
        field_module = self._coordinate_field.getFieldmodule()
        field_cache = field_module.createFieldcache()
        for key_point in self._key_points:
            node = key_point.get_node()
            field_cache.setNode(node)
            time = key_point.get_creation_time()
            field_cache.setTime(time)
            result, coordinates = self._coordinate_field.evaluateReal(field_cache, 3)
            if result == CMISS_OK:
                key_points.append(coordinates)

        return key_points

    def clear(self):
        self._key_points = []
        default_region = self._master_model.get_default_region()
        if self._region is not None:
            default_region.removeChild(self._region)

    def create_model(self):
        self.clear()
        default_region = self._master_model.get_default_region()

        self._region = default_region.createChild('tracking')
        self._coordinate_field = create_finite_element_field(self._region)

        field_module = self._region.getFieldmodule()
        field_module.beginChange()
        self._index_field = field_module.createFieldStoredString()
        self._index_field.setName('index')
        node_set = field_module.findNodesetByName('datapoints')
        # Setup the selection fields
        self._selection_group_field = field_module.createFieldGroup()
        selection_group = self._selection_group_field.createFieldNodeGroup(node_set)
        self._selection_group = selection_group.getNodesetGroup()
        field_module.endChange()


def _get_nearest_match(list_of_numbers, target_number):
    """
    Assumes list_of_numbers is sorted. Returns closest value to target_number.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(list_of_numbers, target_number)
    if pos == 0:
        return list_of_numbers[0]
    if pos == len(list_of_numbers):
        return list_of_numbers[-1]
    before = list_of_numbers[pos - 1]
    after = list_of_numbers[pos]
    if after - target_number < target_number - before:
        return after
    else:
        return before
