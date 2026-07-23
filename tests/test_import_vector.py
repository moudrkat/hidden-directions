"""Vector import from other libraries — pure logic, no torch."""
import json

from hidden_directions.import_vector import to_layer_matrix, load_any


class FakeControlVector:  # mimics repeng's ControlVector.directions
    def __init__(self, directions):
        self.directions = directions


class FakeSteeringVector:  # mimics steering-vectors' layer_activations
    def __init__(self, la):
        self.layer_activations = la


def test_repeng_control_vector():
    cv = FakeControlVector({0: [1.0, 0.0], 2: [0.0, 1.0]})
    M = to_layer_matrix(cv)
    assert len(M) == 3 and M[0] == [1.0, 0.0] and M[1] == [0.0, 0.0] and M[2] == [0.0, 1.0]


def test_steering_vectors_lib():
    sv = FakeSteeringVector({1: [0.5, 0.5]})
    M = to_layer_matrix(sv, n_layers=4)
    assert len(M) == 4 and M[1] == [0.5, 0.5] and M[0] == [0.0, 0.0]


def test_bare_layer_dict():
    M = to_layer_matrix({0: [1.0], 1: [2.0]})
    assert M == [[1.0], [2.0]]


def test_matrix_passthrough():
    assert to_layer_matrix([[1.0, 2.0], [3.0, 4.0]]) == [[1.0, 2.0], [3.0, 4.0]]


def test_single_vector_needs_placement():
    M = to_layer_matrix([1.0, 2.0, 3.0], layer=2, n_layers=5)
    assert len(M) == 5 and M[2] == [1.0, 2.0, 3.0] and M[0] == [0.0, 0.0, 0.0]
    try:
        to_layer_matrix([1.0, 2.0])
        assert False
    except ValueError:
        pass


def test_load_any_json(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"0": [1.0, 2.0]}))
    assert to_layer_matrix(load_any(p)) == [[1.0, 2.0]]
