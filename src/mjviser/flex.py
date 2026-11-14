"""Smooth tube meshes for 1D flexes (ropes and cables)."""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import mujoco
import numpy as np

_TUBE_SIDES = 12
_SAMPLES_PER_EDGE = 8


@dataclasses.dataclass(frozen=True)
class FlexChain:
  """A run of flex vertices joined by edges."""

  vert_ids: np.ndarray
  """Global flex vertex ids, in order along the chain."""
  closed: bool
  """Whether the last vertex connects back to the first."""


def flex_chains(mj_model: mujoco.MjModel, flex_id: int) -> list[FlexChain]:
  """Split the edges of a 1D flex into chains of vertices.

  A flex whose vertices all have at most two neighbors splits into open and
  closed chains. Any other flex falls back to one chain per edge.
  """
  adr = mj_model.flex_elemdataadr[flex_id]
  num_elem = mj_model.flex_elemnum[flex_id]
  edges = mj_model.flex_elem[adr : adr + 2 * num_elem].reshape(num_elem, 2)
  edges = edges + mj_model.flex_vertadr[flex_id]
  neighbors: dict[int, list[int]] = defaultdict(list)
  for a, b in edges:
    neighbors[int(a)].append(int(b))
    neighbors[int(b)].append(int(a))
  if any(len(nbrs) > 2 for nbrs in neighbors.values()):
    return [FlexChain(edge, closed=False) for edge in edges]

  ends = [v for v, nbrs in neighbors.items() if len(nbrs) == 1]
  unvisited = set(neighbors)
  chains: list[FlexChain] = []
  while unvisited:
    # Walk open chains from their ends first; whatever remains is a loop.
    start = next((v for v in ends if v in unvisited), min(unvisited))
    chain = [start]
    unvisited.discard(start)
    while True:
      step = [v for v in neighbors[chain[-1]] if v in unvisited]
      if not step:
        break
      chain.append(step[0])
      unvisited.discard(step[0])
    chains.append(FlexChain(np.array(chain), closed=len(neighbors[start]) == 2))
  return chains


def flex_vertex_positions(
  mj_model: mujoco.MjModel, body_xpos: np.ndarray, body_xmat: np.ndarray
) -> np.ndarray:
  """World positions of all flex vertices from body poses.

  Only valid for flexes without interpolation, whose vertices are bodies.

  Args:
      mj_model: MuJoCo model.
      body_xpos: Body positions, shape ``(num_envs, nbody, 3)``.
      body_xmat: Body rotation matrices, shape ``(num_envs, nbody, 3, 3)``.

  Returns:
      Vertex positions, shape ``(num_envs, nflexvert, 3)``.
  """
  bodies = mj_model.flex_vertbodyid
  offsets = np.einsum("evij,vj->evi", body_xmat[:, bodies], mj_model.flex_vert)
  return body_xpos[:, bodies] + offsets


def _catmull_rom(points: np.ndarray, closed: bool) -> np.ndarray:
  """Sample a uniform Catmull-Rom spline through points of shape (E, N, 3)."""
  if closed:
    padded = np.concatenate([points[:, -1:], points, points[:, :2]], axis=1)
  else:
    head = 2.0 * points[:, :1] - points[:, 1:2]
    tail = 2.0 * points[:, -1:] - points[:, -2:-1]
    padded = np.concatenate([head, points, tail], axis=1)
  p0, p1, p2, p3 = (padded[:, i : padded.shape[1] - 3 + i, None] for i in range(4))
  t = np.linspace(0.0, 1.0, _SAMPLES_PER_EDGE, endpoint=False)[:, None]  # (T, 1)
  samples = 0.5 * (
    2.0 * p1
    + (p2 - p0) * t
    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
    + (3.0 * (p1 - p2) + p3 - p0) * t**3
  )  # (E, S, T, 3)
  samples = samples.reshape(points.shape[0], -1, 3)
  if closed:
    return samples
  return np.concatenate([samples, points[:, -1:]], axis=1)


def _normalize(v: np.ndarray) -> np.ndarray:
  return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _tube_vertices(path: np.ndarray, radius: float, closed: bool) -> np.ndarray:
  """Rings of a tube around paths of shape (E, M, 3), shape (E, M, K, 3)."""
  if closed:
    tangents = np.roll(path, -1, axis=1) - np.roll(path, 1, axis=1)
  else:
    tangents = np.gradient(path, axis=1)
  tangents = _normalize(tangents)

  # Parallel-transport a normal along the path so the rings don't twist.
  first = tangents[:, 0]
  up = np.where(np.abs(first[:, 2:]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
  normal = _normalize(np.cross(first, up))
  normals = np.empty_like(path)
  for i in range(path.shape[1]):
    tangent = tangents[:, i]
    normal = _normalize(normal - np.sum(normal * tangent, -1, keepdims=True) * tangent)
    normals[:, i] = normal
  binormals = np.cross(tangents, normals)

  angles = np.linspace(0.0, 2.0 * np.pi, _TUBE_SIDES, endpoint=False)[:, None]
  ring = (
    np.cos(angles) * normals[:, :, None] + np.sin(angles) * binormals[:, :, None]
  )  # (E, M, K, 3)
  return path[:, :, None] + radius * ring


def _tube_faces(num_rings: int, closed: bool) -> np.ndarray:
  """Outward-facing triangles of one tube, with end caps if open."""
  k = np.arange(_TUBE_SIDES)
  rings = np.arange(num_rings if closed else num_rings - 1)[:, None]
  a = rings * _TUBE_SIDES + k
  b = rings * _TUBE_SIDES + (k + 1) % _TUBE_SIDES
  c = (rings + 1) % num_rings * _TUBE_SIDES + k
  d = (rings + 1) % num_rings * _TUBE_SIDES + (k + 1) % _TUBE_SIDES
  faces = [np.stack([a, b, d], -1), np.stack([a, d, c], -1)]
  if not closed:
    # Cap centers are stored after the rings: start, then end.
    start = num_rings * _TUBE_SIDES
    last = (num_rings - 1) * _TUBE_SIDES
    kn = (k + 1) % _TUBE_SIDES
    faces.append(np.stack([np.full_like(k, start), kn, k], -1))
    faces.append(np.stack([np.full_like(k, start + 1), last + k, last + kn], -1))
  return np.concatenate([f.reshape(-1, 3) for f in faces])


@dataclasses.dataclass(frozen=True)
class FlexTube:
  """Smooth tube through the vertices of a 1D flex."""

  chains: list[FlexChain]
  radius: float

  @staticmethod
  def from_model(mj_model: mujoco.MjModel, flex_id: int) -> FlexTube:
    return FlexTube(
      chains=flex_chains(mj_model, flex_id),
      radius=float(mj_model.flex_radius[flex_id]),
    )

  def mesh(self, vert_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tube vertices and faces for all environments, stacked into one mesh.

    Args:
        vert_pos: Flex vertex positions, shape ``(num_envs, nflexvert, 3)``.

    Returns:
        Vertices of shape ``(V, 3)`` and faces of shape ``(F, 3)``.
    """
    num_envs = vert_pos.shape[0]
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for chain in self.chains:
      path = _catmull_rom(vert_pos[:, chain.vert_ids], chain.closed)  # (E, M, 3)
      verts = _tube_vertices(path, self.radius, chain.closed).reshape(num_envs, -1, 3)
      if not chain.closed:
        verts = np.concatenate([verts, path[:, :1], path[:, -1:]], axis=1)
      chain_faces = _tube_faces(path.shape[1], chain.closed)
      # Every environment's copy of the chain has the same faces, offset.
      env_offsets = offset + verts.shape[1] * np.arange(num_envs)[:, None, None]
      faces.append((chain_faces[None] + env_offsets).reshape(-1, 3))
      vertices.append(verts.reshape(-1, 3))
      offset += verts.shape[0] * verts.shape[1]
    return (
      np.concatenate(vertices).astype(np.float32),
      np.concatenate(faces).astype(np.uint32),
    )
