"""Steerability screen — pure-python path, no torch."""
from hidden_directions.screen import screen_from_sides


def _samples(base, jitter, n=4):
    return [[[b + j * (i % 2 * 2 - 1) for b, j in zip(base, jitter)]] for i in range(n)]


def test_coherent_direction_screens_steerable():
    adv = [[[1.0, 1.0, 0.0]]] * 4
    bal = [[[0.0, 0.0, 0.0]]] * 4
    s = screen_from_sides(adv, bal)
    assert s["verdict"] == "steerable" and s["best_agreement"] > 0.9


def test_incoherent_directions_screen_unsteerable():
    adv = [[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, -1.0, 0.0]]]
    bal = [[[0.0, 0.0, 0.0]]] * 4
    s = screen_from_sides(adv, bal)
    assert s["verdict"] == "unsteerable"


def test_floors_are_reported():
    s = screen_from_sides([[[1.0, 0.0]]] * 2, [[[0.0, 0.0]]] * 2,
                          steerable_floor=0.5, marginal_floor=0.2)
    assert s["floors"] == {"steerable": 0.5, "marginal": 0.2}
