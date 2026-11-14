import mujoco
import numpy as np
import pytest
import viser

from mjviser import ViserMujocoScene
from mjviser.flex import FlexTube, flex_chains, flex_vertex_positions

_ROPE_XML = """
<mujoco>
  <worldbody>
    <body name="anchor" pos="0 0 1">
      <flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                dim="1" radius="0.01" rgba="1 1 1 1">
        <edge equality="true"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
"""

_LOOP_XML = """
<mujoco>
  <worldbody>
    <flexcomp name="loop" type="circle" count="8 1 1" spacing="0.1 0.1 0.1"
              dim="1" radius="0.01"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def rope_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(_ROPE_XML)


def test_rope_is_one_open_chain(rope_model):
  (chain,) = flex_chains(rope_model, 0)
  assert not chain.closed
  np.testing.assert_array_equal(np.sort(chain.vert_ids), np.arange(5))
  assert abs(chain.vert_ids[0] - chain.vert_ids[-1]) == 4


def test_circle_is_one_closed_chain():
  model = mujoco.MjModel.from_xml_string(_LOOP_XML)
  (chain,) = flex_chains(model, 0)
  assert chain.closed
  assert len(chain.vert_ids) == model.flex_vertnum[0]


def test_tube_surrounds_the_rope_in_every_env(rope_model):
  data = mujoco.MjData(rope_model)
  mujoco.mj_kinematics(rope_model, data)
  xpos = np.stack([data.xpos, data.xpos + [0.0, 2.0, 0.0]])
  xmat = np.stack([data.xmat.reshape(-1, 3, 3)] * 2)
  vert_pos = flex_vertex_positions(rope_model, xpos, xmat)  # (2, 5, 3)

  tube = FlexTube.from_model(rope_model, 0)
  vertices, faces = tube.mesh(vert_pos)
  assert faces.max() == len(vertices) - 1
  per_env = vertices.reshape(2, -1, 3)
  for env in range(2):
    # The rope is straight along x, so ring vertices sit one radius off its line.
    rope = vert_pos[env]
    offset = per_env[env] - rope[0]
    off_axis = np.linalg.norm(offset[:, 1:], axis=1)
    on_ring = off_axis > 1e-9
    np.testing.assert_allclose(off_axis[on_ring], tube.radius, atol=1e-6)
    assert np.all(offset[:, 0] >= -1e-6)
    assert np.all(offset[:, 0] <= rope[-1, 0] - rope[0, 0] + 1e-6)


def test_scene_updates_flex_tube_from_body_poses(rope_model):
  server = viser.ViserServer(port=0)
  scene = ViserMujocoScene(server, rope_model, num_envs=2)
  (flex,) = scene._flex_handles
  data = mujoco.MjData(rope_model)
  mujoco.mj_kinematics(rope_model, data)
  xpos = np.stack([data.xpos, data.xpos + [0.0, 2.0, 0.0]])
  xmat = np.stack([data.xmat.reshape(-1, 3, 3)] * 2)

  scene.update_from_arrays(xpos, xmat)
  per_env = flex.handle.vertices.reshape(2, -1, 3)
  np.testing.assert_allclose(per_env[1], per_env[0] + [0.0, 2.0, 0.0], atol=1e-6)

  scene.show_only_selected = True
  scene.update_from_arrays(xpos, xmat, env_idx=1)
  np.testing.assert_allclose(flex.handle.vertices, per_env[1], atol=1e-6)
  server.stop()
