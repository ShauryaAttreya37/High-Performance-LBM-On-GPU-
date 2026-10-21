"""Pull streaming and halfway bounce-back on solid nodes."""

import numpy as np
import pytest

from lbm import lattice
from lbm.boundaries import apply_bounce_back, bounce_back_links, cavity_walls, channel_walls
from lbm.streaming import pull_stream


def test_pull_stream_moves_each_population_along_its_velocity():
    ny, nx, y0, x0 = 5, 6, 2, 3
    f = np.zeros((9, ny, nx))
    f[:, y0, x0] = np.arange(1, 10)
    out = pull_stream(f)
    for i in range(9):
        y, x = y0 + lattice.CY[i], x0 + lattice.CX[i]
        assert out[i, y, x] == i + 1
        assert np.count_nonzero(out[i]) == 1


def test_pull_stream_wraps_periodically():
    f = np.zeros((9, 4, 4))
    f[1, 0, 3] = 1.0  # moving +x from the right edge
    f[4, 0, 0] = 2.0  # moving -y from the bottom edge
    out = pull_stream(f)
    assert out[1, 0, 0] == 1.0
    assert out[4, 3, 0] == 2.0


def test_channel_walls_block_first_and_last_rows():
    solid = channel_walls(nx=5, ny=7)
    assert solid.shape == (7, 5)
    assert solid[0].all() and solid[-1].all()
    assert not solid[1:-1].any()


def test_cavity_walls_close_the_box_and_move_the_lid():
    solid, wall = cavity_walls(n=4, lid_velocity=0.1)
    assert solid.shape == (6, 6)
    assert solid[0].all() and solid[-1].all() and solid[:, 0].all() and solid[:, -1].all()
    assert not solid[1:-1, 1:-1].any()
    np.testing.assert_array_equal(wall[0, -1], 0.1)
    assert not wall[0, :-1].any()
    assert not wall[1].any()


def test_links_mark_populations_pulled_from_solid_neighbours():
    solid = channel_walls(nx=4, ny=5)
    links = bounce_back_links(solid, np.zeros((2, 5, 4)))
    pulled_from_below = {i for i in range(9) if lattice.CY[i] == 1}
    pulled_from_above = {i for i in range(9) if lattice.CY[i] == -1}
    for i in range(9):
        assert links.mask[i, 1].all() == (i in pulled_from_below)
        assert links.mask[i, 3].all() == (i in pulled_from_above)
        assert not links.mask[i, 2].any(), "interior row has no wall links"
        assert not links.mask[i, 0].any() and not links.mask[i, 4].any(), "solid rows never pull"


def test_moving_lid_term_is_six_w_c_dot_u():
    lid = 0.1
    solid, wall = cavity_walls(n=4, lid_velocity=lid)
    links = bounce_back_links(solid, wall)
    top = 4  # last fluid row
    for i in range(9):
        if lattice.CY[i] == -1:
            expected = 6.0 * lattice.W[i] * lattice.CX[i] * lid
            # Interior columns (corners also see the side walls, also moving at the top).
            np.testing.assert_allclose(links.wall_term[i, top, 2:4], expected)
        else:
            assert not links.wall_term[i, top].any()


def test_bounce_back_reflects_the_opposite_post_collision_population():
    solid = channel_walls(nx=3, ny=4)
    links = bounce_back_links(solid, np.zeros((2, 4, 3)))
    rng = np.random.default_rng(0)
    f_post = rng.standard_normal((9, 4, 3))
    streamed = pull_stream(f_post)
    out = apply_bounce_back(streamed, f_post, links)
    for i in range(9):
        expected = np.where(links.mask[i], f_post[lattice.OPPOSITE[i]], streamed[i])
        np.testing.assert_array_equal(out[i], expected)


def test_bounce_back_does_not_modify_inputs():
    solid = channel_walls(nx=3, ny=4)
    links = bounce_back_links(solid, np.zeros((2, 4, 3)))
    f_post = np.ones((9, 4, 3))
    streamed = np.zeros((9, 4, 3))
    apply_bounce_back(streamed, f_post, links)
    assert not streamed.any()


@pytest.mark.parametrize("n", [0, -2])
def test_cavity_rejects_bad_size(n):
    with pytest.raises(ValueError):
        cavity_walls(n=n, lid_velocity=0.1)
