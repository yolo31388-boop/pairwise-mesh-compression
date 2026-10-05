"""
游戏内网格压缩与几何优化系统 - 核心模块

功能：
- compress_mesh:          网格压缩（顶点量化 + 索引压缩）
- geometry_optimization:  几何优化（顶点缓存优化 + 过度绘制优化）
- lod_generation:         LOD生成（误差度量 + 自动生成多级LOD）
- mesh_simplification:    网格简化（边折叠 + 属性保持）
- mesh_merging:           网格合并（材质合并 + 顶点合并）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    pass


def _position(vertex) -> Tuple[float, float, float]:
    """从顶点（元组或字典）中取出位置。"""
    if isinstance(vertex, dict):
        return tuple(vertex["position"])
    return tuple(vertex[:3])


def _attributes(vertex) -> Dict:
    """从顶点中取出除位置外的属性（uv/normal/color 等）。"""
    if isinstance(vertex, dict):
        return {k: v for k, v in vertex.items() if k != "position"}
    return {}


def _make_vertex(position, attributes):
    if attributes:
        v = {"position": tuple(position)}
        v.update(attributes)
        return v
    return tuple(position)


def _lerp_attributes(a: Dict, b: Dict, t: float) -> Dict:
    """对两个顶点的属性做线性插值，保持属性不丢失。"""
    result = dict(a)
    for key, va in a.items():
        if key not in b:
            continue
        vb = b[key]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            result[key] = va * (1.0 - t) + vb * t
        elif isinstance(va, (tuple, list)) and isinstance(vb, (tuple, list)) \
                and len(va) == len(vb):
            result[key] = tuple(x * (1.0 - t) + y * t for x, y in zip(va, vb))
    return result


def _triangles(indices: List[int]) -> List[Tuple[int, int, int]]:
    return [(indices[i], indices[i + 1], indices[i + 2])
            for i in range(0, len(indices) - 2, 3)]


def _flat(tris: List[Tuple[int, int, int]]) -> List[int]:
    out = []
    for t in tris:
        out.extend(t)
    return out


class MeshCompression:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 网格压缩：顶点量化 + 索引压缩
    # ------------------------------------------------------------------
    def compress_mesh(self, vertices: List, indices: List[int],
                      bits: int = 16) -> Dict:
        """压缩网格：顶点位置量化到 bits 位整数，索引做差分编码并选择最小位宽。"""
        positions = [_position(v) for v in vertices]
        if not positions:
            raise ValueError("顶点列表为空")
        mins = [min(p[i] for p in positions) for i in range(3)]
        maxs = [max(p[i] for p in positions) for i in range(3)]
        extents = [max(maxs[i] - mins[i], 1e-12) for i in range(3)]
        max_q = (1 << bits) - 1

        # 顶点量化：浮点位置 -> [0, 2^bits - 1] 整数
        quantized = [
            tuple(int(round((p[i] - mins[i]) / extents[i] * max_q))
                  for i in range(3))
            for p in positions
        ]

        # 索引压缩：差分编码 + 最小位宽
        deltas = []
        prev = 0
        for idx in indices:
            deltas.append(idx - prev)
            prev = idx
        max_index = max(indices) if indices else 0
        if max_index < 256:
            index_format, index_bytes = "uint8", 1
        elif max_index < 65536:
            index_format, index_bytes = "uint16", 2
        else:
            index_format, index_bytes = "uint32", 4

        vertex_bytes = 3 * (bits // 8)
        original_size = len(positions) * 12 + len(indices) * 4
        compressed_size = len(quantized) * vertex_bytes + len(indices) * index_bytes

        result = {
            "quantized_vertices": quantized,
            "bits": bits,
            "min": tuple(mins),
            "scale": tuple(e / max_q for e in extents),
            "compressed_indices": deltas,
            "index_format": index_format,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": (original_size / compressed_size
                                  if compressed_size else 1.0),
        }
        self._history.append(("compress_mesh", time.time()))
        return result

    def decompress_mesh(self, compressed: Dict) -> Tuple[List, List[int]]:
        """解压：反量化顶点并还原差分索引。"""
        bits = compressed["bits"]
        max_q = (1 << bits) - 1
        mins = compressed["min"]
        scale = compressed["scale"]
        vertices = [
            tuple(mins[i] + q[i] * scale[i] for i in range(3))
            for q in compressed["quantized_vertices"]
        ]
        indices = []
        prev = 0
        for d in compressed["compressed_indices"]:
            indices.append(prev + d)
            prev = prev + d
        return vertices, indices

    # ------------------------------------------------------------------
    # 几何优化：顶点缓存优化 + 过度绘制优化
    # ------------------------------------------------------------------
    def geometry_optimization(self, vertices: List, indices: List[int],
                              cache_size: int = 32,
                              view_direction=(0.0, 0.0, -1.0)) -> Dict:
        """优化网格：重排三角形提高顶点缓存命中率，并按深度排序减少过度绘制。"""
        tris = _triangles(indices)
        acmr_before = self._acmr(tris, cache_size)

        # 顶点缓存优化：贪心选择与缓存中顶点重合最多的三角形
        optimized = self._vertex_cache_optimize(tris, cache_size)
        acmr_after = self._acmr(optimized, cache_size)

        # 过度绘制优化：按观察方向深度从前到后排序，利于 early-z 剔除
        positions = [_position(v) for v in vertices]
        d = view_direction
        optimized.sort(key=lambda t: sum(
            positions[i][0] * d[0] + positions[i][1] * d[1] + positions[i][2] * d[2]
            for i in t) / 3.0)

        result = {
            "indices": _flat(optimized),
            "triangle_count": len(optimized),
            "acmr_before": acmr_before,
            "acmr_after": acmr_after,
            "cache_size": cache_size,
            "overdraw_sorted": True,
        }
        self._history.append(("geometry_optimization", time.time()))
        return result

    @staticmethod
    def _acmr(tris: List[Tuple[int, int, int]], cache_size: int) -> float:
        """平均缓存未命中率（ACMR）。"""
        if not tris:
            return 0.0
        cache: List[int] = []
        misses = 0
        for tri in tris:
            for v in tri:
                if v in cache:
                    cache.remove(v)
                else:
                    misses += 1
                cache.append(v)
                if len(cache) > cache_size:
                    cache.pop(0)
        return misses / len(tris)

    @staticmethod
    def _vertex_cache_optimize(tris: List[Tuple[int, int, int]],
                               cache_size: int) -> List[Tuple[int, int, int]]:
        """贪心顶点缓存优化：每次优先选择与最近使用顶点重合最多的三角形。"""
        remaining = list(tris)
        optimized: List[Tuple[int, int, int]] = []
        cache: List[int] = []
        while remaining:
            best_i, best_score = 0, -1
            limit = min(len(remaining), 64)
            for i in range(limit):
                score = sum(1 for v in remaining[i] if v in cache)
                if score > best_score:
                    best_i, best_score = i, score
                if score == 3:
                    break
            tri = remaining.pop(best_i)
            optimized.append(tri)
            for v in tri:
                if v in cache:
                    cache.remove(v)
                cache.append(v)
                if len(cache) > cache_size:
                    cache.pop(0)
        return optimized

    # ------------------------------------------------------------------
    # LOD生成：误差度量 + 自动生成多级LOD
    # ------------------------------------------------------------------
    def lod_generation(self, vertices: List, indices: List[int],
                       levels: int = 3, ratio: float = 0.5) -> List[Dict]:
        """自动生成多个LOD级别，每级附带几何误差度量。"""
        if levels < 1:
            raise ValueError("levels 必须 >= 1")
        lods = [{
            "level": 0,
            "vertices": list(vertices),
            "indices": list(indices),
            "error": 0.0,
            "triangle_count": len(indices) // 3,
        }]
        cur_vertices, cur_indices = list(vertices), list(indices)
        cumulative_error = 0.0
        for level in range(1, levels):
            simplified = self.mesh_simplification(cur_vertices, cur_indices,
                                                  target_ratio=ratio)
            cur_vertices = simplified["vertices"]
            cur_indices = simplified["indices"]
            cumulative_error += simplified["error"]
            lods.append({
                "level": level,
                "vertices": cur_vertices,
                "indices": cur_indices,
                "error": cumulative_error,
                "triangle_count": len(cur_indices) // 3,
            })
        self._history.append(("lod_generation", time.time()))
        return lods

    # ------------------------------------------------------------------
    # 网格简化：边折叠 + 属性保持
    # ------------------------------------------------------------------
    def mesh_simplification(self, vertices: List, indices: List[int],
                            target_ratio: float = 0.5) -> Dict:
        """用边折叠算法简化网格，折叠时对顶点属性做插值以保持属性。"""
        positions = [_position(v) for v in vertices]
        attrs = [_attributes(v) for v in vertices]
        tris = [list(t) for t in _triangles(indices)]
        target = max(1, int(round(len(tris) * target_ratio)))
        max_error = 0.0
        collapsed = 0

        alive = [True] * len(positions)
        while len(tris) > target:
            # 找最短的边做折叠
            best = None
            best_len = float("inf")
            seen = set()
            for tri in tris:
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]),
                             (tri[2], tri[0])):
                    key = (a, b) if a < b else (b, a)
                    if key in seen:
                        continue
                    seen.add(key)
                    pa, pb = positions[a], positions[b]
                    length = math.dist(pa, pb)
                    if length < best_len:
                        best_len = length
                        best = (a, b)
            if best is None:
                break
            a, b = best
            # 边折叠：b 合并到 a，位置取中点，属性线性插值（属性保持）
            positions[a] = tuple((positions[a][i] + positions[b][i]) * 0.5
                                 for i in range(3))
            attrs[a] = _lerp_attributes(attrs[a], attrs[b], 0.5)
            alive[b] = False
            collapsed += 1
            max_error = max(max_error, best_len * 0.5)
            new_tris = []
            for tri in tris:
                tri = [a if v == b else v for v in tri]
                if len(set(tri)) == 3:
                    new_tris.append(tri)
            tris = new_tris

        # 压缩顶点数组并重映射索引
        remap = {}
        out_vertices = []
        for i, is_alive in enumerate(alive):
            if is_alive and any(i in t for t in tris):
                remap[i] = len(out_vertices)
                out_vertices.append(_make_vertex(positions[i], attrs[i]))
        out_indices = [remap[v] for tri in tris for v in tri]

        result = {
            "vertices": out_vertices,
            "indices": out_indices,
            "collapsed_edges": collapsed,
            "error": max_error,
            "triangle_count": len(out_indices) // 3,
        }
        self._history.append(("mesh_simplification", time.time()))
        return result

    # ------------------------------------------------------------------
    # 网格合并：材质合并 + 顶点合并
    # ------------------------------------------------------------------
    def mesh_merging(self, meshes: List[Dict],
                     weld_epsilon: float = 1e-6) -> List[Dict]:
        """合并网格：相同材质合并为一个网格，重合顶点焊接为一个顶点。"""
        groups: Dict[str, List[Dict]] = {}
        for mesh in meshes:
            material = mesh.get("material", "default")
            groups.setdefault(material, []).append(mesh)

        merged = []
        for material, group in groups.items():
            out_vertices: List = []
            out_indices: List[int] = []
            weld_map: Dict[Tuple[int, int, int], int] = {}
            welded = 0
            for mesh in group:
                verts = mesh["vertices"]
                remap = {}
                for i, v in enumerate(verts):
                    p = _position(v)
                    key = (round(p[0] / weld_epsilon),
                           round(p[1] / weld_epsilon),
                           round(p[2] / weld_epsilon))
                    if key in weld_map:
                        remap[i] = weld_map[key]
                        welded += 1
                    else:
                        remap[i] = len(out_vertices)
                        weld_map[key] = remap[i]
                        out_vertices.append(v)
                out_indices.extend(remap[i] for i in mesh["indices"])
            merged.append({
                "material": material,
                "vertices": out_vertices,
                "indices": out_indices,
                "welded_vertices": welded,
                "source_meshes": len(group),
            })
        self._history.append(("mesh_merging", time.time()))
        return merged
