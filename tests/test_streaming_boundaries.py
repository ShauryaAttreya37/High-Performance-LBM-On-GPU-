"""Pull streaming and halfway bounce-back on solid nodes, in 2D and 3D."""

import numpy as np
import pytest

from lbm.boundaries import apply_bounce_back, bounce_back_links, cavity_walls, channel_walls
from lbm.lattice import D2Q9, D3Q19
from lbm.streaming import pull_stream
from tests.helpers import ALL_LATTICES

GRIDS = {2: (5, 6), 3: (4, 5, 6)}


def index_offset(c):
    """Array-index offset of velocity c: arrays are [y, x] / [z, y, x], so reverse it."""
    return tuple(reversed(c))


def channel(vs):
    """Channel mask with walls at y = 0 and y = ny - 1 (2D channel or 3D slab)."""
    shape = GRIDS[vs.d]
    nz = shape[0] if vs.d == 3 else None
    return channel_walls(nx=shape[-1], ny=shape[-2], nz=nz)


@ALL_LATTICES
def test_pull_stream_moves_each_population_along_its_velocity(vs):
    shape = GRIDS[vs.d]
    centre = tuple(n // 2 for n in shape)
    f = np.zeros((vs.q, *shape))
    for i in range(vs.q):
        f[(i, *centre)] = i + 1
    out = pull_stream(vs, f)
    for i, c in enumerate(vs.c):
        target = tuple(p + o for p, o in zip(centre, index_offset(c), strict=True))
        assert out[(i, *target)] == i + 1
        assert np.count_nonzero(out[i]) == 1


@ALL_LATTICES
def test_pull_stream_wraps_periodically(vs):
    shape = GRIDS[vs.d]
    i = vs.c.index((1,) + (0,) * (vs.d - 1))  # +x
    f = np.zeros((vs.q, *shape))
    f[(i, *(0,) * (vs.d - 1), shape[-1] - 1)] = 1.0  # at the right edge
    out = pull_stream(vs, f)
    assert out[(i, *(0,) * vs.d)] == 1.0


def test_channel_walls_block_first_and_last_rows_in_2d_and_3d():
    solid = channel_walls(nx=5, ny=7)
    assert solid.shape == (7, 5)
    assert solid[0].all() and solid[-1].all() and not solid[1:-1].any()
    slab = channel_walls(nx=5, ny=7, nz=3)
    assert slab.shape == (3, 7, 5)
    assert slab[:, 0].all() and slab[:, -1].all() and not slab[:, 1:-1].any()


def test_cavity_walls_close_the_box_and_move_the_lid_2d():
    solid, wall = cavity_walls(n=4, lid_velocity=0.1)
    assert solid.shape == (6, 6) and wall.shape == (2, 6, 6)
    assert solid[0].all() and solid[-1].all() and solid[:, 0].all() and solid[:, -1].all()
    assert not solid[1:-1, 1:-1].any()
    np.testing.assert_array_equal(wall[0, -1], 0.1)
    assert not wall[0, :-1].any() and not wall[1].any()


def test_cavity_walls_close_the_box_and_move_the_lid_3d():
    solid, wall = cavity_walls(n=4, lid_velocity=0.1, dimensions=3)
    assert solid.shape == (6, 6, 6) and wall.shape == (3, 6, 6, 6)
    assert not solid[1:-1, 1:-1, 1:-1].any()
    assert solid.sum() == 6**3 - 4**3
    np.testing.assert_array_equal(wall[0, :, -1, :], 0.1)  # top face y = n + 1 moves in +x
    assert not wall[0, :, :-1, :].any() and not wall[1:].any()


@ALL_LATTICES
def test_links_mark_populations_pulled_from_solid_neighbours(vs):
    solid = channel(vs)
    links = bounce_back_links(vs, solid, np.zeros((vs.d, *solid.shape)))
    ny = solid.shape[-2]
    for i, c in enumerate(vs.c):
        rows = links.mask[i].swapaxes(0, -2) if vs.d == 3 else links.mask[i]
        assert rows[1].all() == (c[1] == 1), "first fluid row pulls c_y = +1 from below"
        assert rows[ny - 2].all() == (c[1] == -1), "last fluid row pulls c_y = -1 from above"
        assert not rows[2 : ny - 2].any(), "interior rows have no wall links"
        assert not rows[0].any() and not rows[ny - 1].any(), "solid rows never pull"


@pytest.mark.parametrize(("vs", "dimensions"), [(D2Q9, 2), (D3Q19, 3)], ids=["D2Q9", "D3Q19"])
def test_moving_lid_term_is_two_w_c_dot_u_over_cs2(vs, dimensions):
    lid, n = 0.1, 4
    solid, wall = cavity_walls(n=n, lid_velocity=lid, dimensions=dimensions)
    links = bounce_back_links(vs, solid, wall)
    for i, c in enumerate(vs.c):
        term = links.wall_term[i]
        interior = term[..., n, 2:4] if dimensions == 2 else term[2:4, n, 2:4]  # top fluid row
        if c[1] == -1:  # pulled from the lid above
            np.testing.assert_allclose(interior, 6.0 * vs.w[i] * c[0] * lid)
        else:
            assert not interior.any()


@ALL_LATTICES
def test_bounce_back_reflects_the_opposite_post_collision_population(vs):
    solid = channel(vs)
    links = bounce_back_links(vs, solid, np.zeros((vs.d, *solid.shape)))
    f_post = np.random.default_rng(0).standard_normal((vs.q, *solid.shape))
    streamed = pull_stream(vs, f_post)
    out = apply_bounce_back(vs, streamed, f_post, links)
    for i in range(vs.q):
        expected = np.where(links.mask[i], f_post[vs.opposite[i]], streamed[i])
        np.testing.assert_array_equal(out[i], expected)


@ALL_LATTICES
def test_bounce_back_does_not_modify_inputs(vs):
    solid = channel(vs)
    links = bounce_back_links(vs, solid, np.zeros((vs.d, *solid.shape)))
    f_post = np.ones((vs.q, *solid.shape))
    streamed = np.zeros((vs.q, *solid.shape))
    apply_bounce_back(vs, streamed, f_post, links)
    assert not streamed.any() and np.all(f_post == 1.0)


@pytest.mark.parametrize("n", [0, -2])
@pytest.mark.parametrize("dimensions", [2, 3])
def test_cavity_rejects_bad_size(n, dimensions):
    with pytest.raises(ValueError, match="size"):
        cavity_walls(n=n, lid_velocity=0.1, dimensions=dimensions)


def test_cavity_rejects_unsupported_dimension():
    with pytest.raises(ValueError, match="dimensions"):
        cavity_walls(n=4, lid_velocity=0.1, dimensions=4)


def test_channel_rejects_bad_size():
    with pytest.raises(ValueError, match="channel"):
        channel_walls(nx=4, ny=2)
