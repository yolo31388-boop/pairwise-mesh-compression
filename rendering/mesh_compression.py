"""
游戏内网格压缩与几何优化系统 - 核心模块

功能：
- compress_mesh:        网格压缩（顶点量化 + 索引压缩）
- geometry_optimization: 几何优化（顶点缓存优化 + 过度绘制优化）
- lod_generation:       LOD 生成（误差度量 + 自动生成多级 LOD）
- mesh_simplification:  网格简化（边折叠 + 属性保持）
- mesh_merging:         网格合并（材质合并 + 顶点合并）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    pass


def _dist(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


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

    def _record(self, operation: str, **info):
        self._history.append({"operation": operation, "time": time.time(), **info})

    # ------------------------------------------------------------------
    # 网格压缩：顶点量化 + 索引压缩
    # ------------------------------------------------------------------
    def compress_mesh(self, vertices: List[Tuple[float, float, float]],
                      indices: List[int],
                      quantization_bits: int = 16) -> Dict:
        if not 4 <= quantization_bits <= 32:
            raise ValueError("quantization_bits 必须在 4~32 之间")
        vertices = [tuple(map(float, v)) for v in vertices]
        indices = list(indices)
        levels = (1 << quantization_bits) - 1

        # --- 顶点量化：按包围盒把浮点位置映射到 N 位整数 ---
        bbox_min = [0.0, 0.0, 0.0]
        bbox_max = [0.0, 0.0, 0.0]
        if vertices:
            for axis in range(3):
                bbox_min[axis] = min(v[axis] for v in vertices)
                bbox_max[axis] = max(v[axis] for v in vertices)
        scale = [max(bbox_max[a] - bbox_min[a], 1e-12) for a in range(3)]
        quantized = [
            tuple(int(round((v[a] - bbox_min[a]) / scale[a] * levels)) for a in range(3))
            for v in vertices
        ]

        # --- 索引压缩：差分编码 + zigzag + 变长字节编码 ---
        compressed_indices = self._encode_indices(indices)

        bytes_per_component = (quantization_bits + 7) // 8
        original_size = len(vertices) * 3 * 4 + len(indices) * 4
        compressed_size = len(quantized) * 3 * bytes_per_component + len(compressed_indices)
        ratio = (original_size / compressed_size) if compressed_size else 1.0

        result = {
            "quantized_vertices": quantized,
            "compressed_indices": compressed_indices,
            "quantization_bits": quantization_bits,
            "bbox_min": tuple(bbox_min),
            "bbox_max": tuple(bbox_max),
            "vertex_count": len(vertices),
            "index_count": len(indices),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": ratio,
        }
        self._record("compress_mesh", vertices=len(vertices), ratio=ratio)
        return result

    def decompress_mesh(self, compressed: Dict) -> Dict:
        """解压：反量化顶点 + 解码索引，用于校验压缩结果。"""
        bits = compressed["quantization_bits"]
        levels = (1 << bits) - 1
        bbox_min = compressed["bbox_min"]
        bbox_max = compressed["bbox_max"]
        scale = [max(bbox_max[a] - bbox_min[a], 1e-12) for a in range(3)]
        vertices = [
            tuple(bbox_min[a] + q[a] / levels * scale[a] for a in range(3))
            for q in compressed["quantized_vertices"]
        ]
        indices = self._decode_indices(compressed["compressed_indices"],
                                       compressed["index_count"])
        return {"vertices": vertices, "indices": indices}

    @staticmethod
    def _encode_indices(indices: List[int]) -> bytes:
        out = bytearray()
        prev = 0
        for idx in indices:
            delta = idx - prev
            prev = idx
            zz = (delta << 1) if delta >= 0 else (((-delta) << 1) - 1)
            while True:
                byte = zz & 0x7F
                zz >>= 7
                if zz:
                    out.append(byte | 0x80)
                else:
                    out.append(byte)
                    break
        return bytes(out)

    @staticmethod
    def _decode_indices(data: bytes, count: int) -> List[int]:
        indices = []
        prev = 0
        shift = 0
        value = 0
        for byte in data:
            value |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                delta = (value >> 1) if (value & 1) == 0 else -((value + 1) >> 1)
                prev += delta
                indices.append(prev)
                value = 0
                shift = 0
                if len(indices) == count:
                    break
        return indices

    # ------------------------------------------------------------------
    # 几何优化：顶点缓存优化 + 过度绘制优化
    # ------------------------------------------------------------------
    def geometry_optimization(self, vertices: List[Tuple[float, float, float]],
                              indices: List[int],
                              cache_size: int = 32) -> Dict:
        vertices = [tuple(map(float, v)) for v in vertices]
        triangles = [tuple(indices[i:i + 3]) for i in range(0, len(indices) - 2, 3)]

        hit_rate_before = self._cache_hit_rate(indices, cache_size)
        overdraw_before = self._estimate_overdraw(vertices, triangles)

        # --- 顶点缓存优化：贪心重排三角形，最大化缓存命中 ---
        triangles = self._optimize_vertex_cache(triangles, cache_size)

        # --- 过度绘制优化：按质心深度从前到后排序 ---
        triangles.sort(key=lambda t: sum(vertices[v][2] for v in t) / 3.0)

        optimized_indices = [v for tri in triangles for v in tri]
        hit_rate_after = self._cache_hit_rate(optimized_indices, cache_size)
        overdraw_after = self._estimate_overdraw(vertices, triangles)

        result = {
            "vertices": vertices,
            "indices": optimized_indices,
            "triangle_count": len(triangles),
            "cache_hit_rate_before": hit_rate_before,
            "cache_hit_rate_after": hit_rate_after,
            "overdraw_before": overdraw_before,
            "overdraw_after": overdraw_after,
        }
        self._record("geometry_optimization", triangles=len(triangles))
        return result

    @staticmethod
    def _cache_hit_rate(indices: List[int], cache_size: int) -> float:
        if not indices:
            return 0.0
        cache: List[int] = []
        hits = 0
        for v in indices:
            if v in cache:
                hits += 1
                cache.remove(v)
            cache.insert(0, v)
            del cache[cache_size:]
        return hits / len(indices)

    @staticmethod
    def _optimize_vertex_cache(triangles: List[Tuple[int, int, int]],
                               cache_size: int) -> List[Tuple[int, int, int]]:
        remaining = list(triangles)
        ordered: List[Tuple[int, int, int]] = []
        cache: List[int] = []
        while remaining:
            best_pos, best_score = 0, -1
            for pos, tri in enumerate(remaining):
                score = sum(1 for v in tri if v in cache)
                if score > best_score:
                    best_pos, best_score = pos, score
                    if score == 3:
                        break
            tri = remaining.pop(best_pos)
            ordered.append(tri)
            for v in tri:
                if v in cache:
                    cache.remove(v)
                cache.insert(0, v)
            del cache[cache_size:]
        return ordered

    @staticmethod
    def _estimate_overdraw(vertices, triangles) -> int:
        """估算过度绘制：屏幕空间包围盒重叠、且后画的三角形更近（先画的着色被覆盖浪费）。"""
        boxes = []
        depths = []
        for tri in triangles:
            xs = [vertices[v][0] for v in tri]
            ys = [vertices[v][1] for v in tri]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
            depths.append(sum(vertices[v][2] for v in tri) / 3.0)
        overdraw = 0
        for j in range(len(triangles)):
            for i in range(j):
                a, b = boxes[i], boxes[j]
                overlap = (a[0] < b[2] and b[0] < a[2] and
                           a[1] < b[3] and b[1] < a[3])
                if overlap and depths[j] < depths[i]:
                    overdraw += 1
        return overdraw

    # ------------------------------------------------------------------
    # LOD 生成：误差度量 + 自动生成多个级别
    # ------------------------------------------------------------------
    def lod_generation(self, vertices: List[Tuple[float, float, float]],
                       indices: List[int],
                       num_levels: int = 3,
                       reduction_ratio: float = 0.5,
                       attributes: Optional[Dict[str, list]] = None) -> List[Dict]:
        if num_levels < 1:
            raise ValueError("num_levels 必须 >= 1")
        if not 0.0 < reduction_ratio <= 1.0:
            raise ValueError("reduction_ratio 必须在 (0, 1] 之间")

        current_vertices = [tuple(map(float, v)) for v in vertices]
        current_indices = list(indices)
        current_attrs = {k: list(v) for k, v in (attributes or {}).items()} or None

        levels = [{
            "level": 0,
            "vertices": current_vertices,
            "indices": current_indices,
            "attributes": current_attrs,
            "triangle_count": len(current_indices) // 3,
            "error": 0.0,
            "cumulative_error": 0.0,
        }]

        cumulative_error = 0.0
        for level in range(1, num_levels):
            simplified = self.mesh_simplification(
                current_vertices, current_indices,
                target_ratio=reduction_ratio,
                attributes=current_attrs,
            )
            cumulative_error += simplified["error"]
            current_vertices = simplified["vertices"]
            current_indices = simplified["indices"]
            current_attrs = simplified["attributes"]
            levels.append({
                "level": level,
                "vertices": current_vertices,
                "indices": current_indices,
                "attributes": current_attrs,
                "triangle_count": len(current_indices) // 3,
                "error": simplified["error"],
                "cumulative_error": cumulative_error,
            })

        self._record("lod_generation", levels=num_levels)
        return levels

    # ------------------------------------------------------------------
    # 网格简化：边折叠 + 属性保持
    # ------------------------------------------------------------------
    def mesh_simplification(self, vertices: List[Tuple[float, float, float]],
                            indices: List[int],
                            target_ratio: float = 0.5,
                            target_count: Optional[int] = None,
                            attributes: Optional[Dict[str, list]] = None) -> Dict:
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError("target_ratio 必须在 (0, 1] 之间")
        verts = [tuple(map(float, v)) for v in vertices]
        tris = [tuple(indices[i:i + 3]) for i in range(0, len(indices) - 2, 3)]
        attrs = {k: list(v) for k, v in (attributes or {}).items()}
        for name, values in attrs.items():
            if len(values) != len(verts):
                raise ValueError(f"属性 {name} 的数量必须与顶点数一致")

        if target_count is None:
            target_count = max(1, int(round(len(tris) * target_ratio)))
        target_count = max(1, min(target_count, len(tris)))

        collapsed_edges = 0
        max_error = 0.0
        while len(tris) > target_count:
            edge = self._find_shortest_edge(verts, tris)
            if edge is None:
                break
            keep, drop, length = edge
            # 边折叠：把 drop 折叠到 keep，新位置取中点
            verts[keep] = tuple((verts[keep][a] + verts[drop][a]) / 2.0 for a in range(3))
            # 属性保持：对顶点属性做插值
            for name, values in attrs.items():
                values[keep] = self._lerp_attribute(values[keep], values[drop], 0.5)
            tris = [tuple(keep if v == drop else v for v in tri) for tri in tris]
            tris = [tri for tri in tris if len(set(tri)) == 3]
            collapsed_edges += 1
            max_error = max(max_error, length / 2.0)

        # 压缩顶点数组并重映射索引
        used = sorted({v for tri in tris for v in tri})
        remap = {old: new for new, old in enumerate(used)}
        new_vertices = [verts[i] for i in used]
        new_attrs = {name: [values[i] for i in used] for name, values in attrs.items()}
        new_indices = [remap[v] for tri in tris for v in tri]

        result = {
            "vertices": new_vertices,
            "indices": new_indices,
            "attributes": new_attrs,
            "triangle_count": len(tris),
            "collapsed_edges": collapsed_edges,
            "error": max_error,
        }
        self._record("mesh_simplification", collapsed=collapsed_edges)
        return result

    @staticmethod
    def _find_shortest_edge(verts, tris):
        best = None
        best_len = math.inf
        for tri in tris:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                length = _dist(verts[a], verts[b])
                if length < best_len:
                    best_len = length
                    best = (a, b, length)
        return best

    @staticmethod
    def _lerp_attribute(a, b, t):
        if isinstance(a, (tuple, list)):
            return tuple(x * (1 - t) + y * t for x, y in zip(a, b))
        return a * (1 - t) + b * t

    # ------------------------------------------------------------------
    # 网格合并：材质合并 + 顶点合并
    # ------------------------------------------------------------------
    def mesh_merging(self, meshes: List[Dict],
                     vertex_epsilon: float = 1e-6) -> Dict:
        if not meshes:
            raise ValueError("meshes 不能为空")

        # --- 材质合并：相同材质的网格归入同一子网格 ---
        material_order: List[str] = []
        by_material: Dict[str, List[Dict]] = {}
        for mesh in meshes:
            material = mesh.get("material", "default")
            if material not in by_material:
                by_material[material] = []
                material_order.append(material)
            by_material[material].append(mesh)

        merged_vertices: List[Tuple[float, float, float]] = []
        merged_indices: List[int] = []
        submeshes = []
        original_vertex_count = 0

        for material in material_order:
            # 顶点合并：同一材质组内重合的顶点只保留一份
            vertex_map: Dict[Tuple[int, ...], int] = {}
            index_start = len(merged_indices)
            for mesh in by_material[material]:
                verts = mesh["vertices"]
                idxs = mesh["indices"]
                original_vertex_count += len(verts)
                local_map: Dict[int, int] = {}
                for i, v in enumerate(verts):
                    key = tuple(int(round(c / vertex_epsilon)) for c in v)
                    if key not in vertex_map:
                        vertex_map[key] = len(merged_vertices)
                        merged_vertices.append(tuple(map(float, v)))
                    local_map[i] = vertex_map[key]
                merged_indices.extend(local_map[i] for i in idxs)
            submeshes.append({
                "material": material,
                "index_start": index_start,
                "index_count": len(merged_indices) - index_start,
            })

        result = {
            "vertices": merged_vertices,
            "indices": merged_indices,
            "submeshes": submeshes,
            "materials": material_order,
            "original_vertex_count": original_vertex_count,
            "merged_vertex_count": len(merged_vertices),
            "triangle_count": len(merged_indices) // 3,
        }
        self._record("mesh_merging",
                     meshes=len(meshes), materials=len(material_order))
        return result
