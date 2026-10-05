"""
游戏内网格压缩与几何优化系统 - 验收测试
运行方式：python -m pytest tests/test_mesh_compression.py -q
共 15 个测试用例
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.mesh_compression import MeshCompression


def grid_mesh(n=8):
    """生成 n x n 网格平面（共享顶点的规则网格）。"""
    vertices = []
    indices = []
    for j in range(n):
        for i in range(n):
            vertices.append((float(i), float(j), 0.0))
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            indices += [a, a + 1, a + n, a + 1, a + n + 1, a + n]
    return vertices, indices


class TestMeshCompression:
    def setup_method(self):
        self.system = MeshCompression()

    # ---- 网格压缩：顶点量化 ----
    def test_case_01(self):
        """compress_mesh: 顶点量化为指定位宽的整数。"""
        vertices, indices = grid_mesh(4)
        compressed = self.system.compress_mesh(vertices, indices, bits=16)
        assert len(compressed["quantized_vertices"]) == len(vertices)
        for q in compressed["quantized_vertices"]:
            for c in q:
                assert isinstance(c, int)
                assert 0 <= c <= 65535
        assert compressed["quantized_vertices"][0] == (0, 0, 0)
        assert compressed["quantized_vertices"][-1] == (65535, 65535, 0)

    # ---- 网格压缩：索引压缩（差分+最小位宽+可还原）----
    def test_case_02(self):
        """compress_mesh: 索引做差分编码并选择最小位宽，可无损还原。"""
        vertices, indices = grid_mesh(4)
        compressed = self.system.compress_mesh(vertices, indices)
        assert compressed["index_format"] in ("uint8", "uint16", "uint32")
        assert compressed["index_format"] == "uint8"
        assert len(compressed["compressed_indices"]) == len(indices)
        _, restored = self.system.decompress_mesh(compressed)
        assert restored == indices

    # ---- 网格压缩：确实变小 ----
    def test_case_03(self):
        """compress_mesh: 压缩后体积小于原始未压缩数据。"""
        vertices, indices = grid_mesh(16)
        compressed = self.system.compress_mesh(vertices, indices, bits=8)
        assert compressed["compressed_size"] < compressed["original_size"]
        assert compressed["compression_ratio"] > 1.0

    # ---- 几何优化：顶点缓存优化 ----
    def test_case_04(self):
        """geometry_optimization: 重排后顶点缓存未命中率（ACMR）不升高。"""
        vertices, indices = grid_mesh(12)
        scrambled = list(indices)
        import random
        random.Random(1).shuffle(scrambled)
        result = self.system.geometry_optimization(vertices, scrambled)
        assert result["acmr_after"] <= result["acmr_before"]
        assert result["acmr_after"] < 1.5

    # ---- 几何优化：过度绘制优化 ----
    def test_case_05(self):
        """geometry_optimization: 三角形按深度从前到后排序以减少过度绘制。"""
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                    (0, 0, 5), (1, 0, 5), (0, 1, 5),
                    (0, 0, 2), (1, 0, 2), (0, 1, 2)]
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        result = self.system.geometry_optimization(vertices, indices)
        assert result["overdraw_sorted"] is True
        opt = result["indices"]
        depths = []
        for k in range(0, len(opt), 3):
            depths.append(sum(vertices[opt[k + c]][2] for c in range(3)) / 3.0)
        assert depths == sorted(depths, reverse=True)

    # ---- 几何优化：保持几何不变 ----
    def test_case_06(self):
        """geometry_optimization: 三角形数量、索引范围保持一致。"""
        vertices, indices = grid_mesh(8)
        result = self.system.geometry_optimization(vertices, indices)
        assert result["triangle_count"] == len(indices) // 3
        assert len(result["indices"]) == len(indices)
        assert sorted(result["indices"]) == sorted(indices)

    # ---- LOD生成：自动生成多个级别 ----
    def test_case_07(self):
        """lod_generation: 自动生成指定数量的LOD级别。"""
        vertices, indices = grid_mesh(16)
        lods = self.system.lod_generation(vertices, indices, levels=4)
        assert len(lods) == 4
        assert [lod["level"] for lod in lods] == [0, 1, 2, 3]

    # ---- LOD生成：误差度量 ----
    def test_case_08(self):
        """lod_generation: 每级带误差度量，且累积误差随级别非递减。"""
        vertices, indices = grid_mesh(16)
        lods = self.system.lod_generation(vertices, indices, levels=4)
        errors = [lod["error"] for lod in lods]
        assert errors[0] == 0.0
        for e in errors[1:]:
            assert e >= 0.0
            assert math.isfinite(e)
        assert errors[-1] > errors[0]
        assert errors == sorted(errors)

    # ---- LOD生成：三角形数逐级减少 ----
    def test_case_09(self):
        """lod_generation: 越远的LOD三角形越少。"""
        vertices, indices = grid_mesh(16)
        lods = self.system.lod_generation(vertices, indices, levels=3, ratio=0.5)
        counts = [lod["triangle_count"] for lod in lods]
        assert counts[0] > counts[1] > counts[2]
        for lod in lods:
            assert len(lod["indices"]) == lod["triangle_count"] * 3

    # ---- 网格简化：边折叠 ----
    def test_case_10(self):
        """mesh_simplification: 边折叠算法实际减少三角形和顶点。"""
        vertices, indices = grid_mesh(16)
        result = self.system.mesh_simplification(vertices, indices, target_ratio=0.25)
        assert result["collapsed_edges"] > 0
        assert result["triangle_count"] < len(indices) // 3
        assert len(result["vertices"]) < len(vertices)
        assert result["triangle_count"] >= 1

    # ---- 网格简化：属性保持 ----
    def test_case_11(self):
        """mesh_simplification: 折叠后顶点属性（uv）仍然保留。"""
        vertices, indices = grid_mesh(8)
        rich = [{"position": p, "uv": (p[0] / 8.0, p[1] / 8.0)} for p in vertices]
        result = self.system.mesh_simplification(rich, indices, target_ratio=0.5)
        assert result["triangle_count"] > 0
        for v in result["vertices"]:
            assert "uv" in v
            u, w = v["uv"]
            assert math.isfinite(u) and math.isfinite(w)

    # ---- 网格简化：输出合法 ----
    def test_case_12(self):
        """mesh_simplification: 输出索引合法且无退化三角形。"""
        vertices, indices = grid_mesh(12)
        result = self.system.mesh_simplification(vertices, indices, target_ratio=0.4)
        n = len(result["vertices"])
        inds = result["indices"]
        assert all(0 <= i < n for i in inds)
        for k in range(0, len(inds), 3):
            assert len(set(inds[k:k + 3])) == 3

    # ---- 网格合并：材质合并 ----
    def test_case_13(self):
        """mesh_merging: 相同材质的多个网格合并为一个。"""
        v1, i1 = grid_mesh(4)
        v2, i2 = grid_mesh(3)
        meshes = [
            {"material": "wood", "vertices": v1, "indices": i1},
            {"material": "wood", "vertices": [(p[0] + 10, p[1] + 10, p[2]) for p in v2],
             "indices": i2},
            {"material": "metal", "vertices": [(p[0] + 20, p[1] + 20, p[2]) for p in v2],
             "indices": i2},
        ]
        result = self.system.mesh_merging(meshes)
        materials = sorted(m["material"] for m in result)
        assert materials == ["metal", "wood"]
        wood = next(m for m in result if m["material"] == "wood")
        assert wood["source_meshes"] == 2
        assert len(wood["vertices"]) == len(v1) + len(v2)
        assert len(wood["indices"]) == len(i1) + len(i2)

    # ---- 网格合并：顶点合并（焊接重合顶点）----
    def test_case_14(self):
        """mesh_merging: 重合顶点被焊接为一个顶点。"""
        a = {"position": (0.0, 0.0, 0.0)}
        b = {"position": (1.0, 0.0, 0.0)}
        c_dup = {"position": (0.0, 0.0, 0.0)}
        meshes = [
            {"material": "x", "vertices": [a, b], "indices": [0, 1]},
            {"material": "x", "vertices": [c_dup, b], "indices": [0, 1]},
        ]
        result = self.system.mesh_merging(meshes)
        assert len(result) == 1
        merged = result[0]
        assert len(merged["vertices"]) == 2
        assert merged["welded_vertices"] >= 2
        assert all(0 <= i < len(merged["vertices"]) for i in merged["indices"])

    # ---- 网格合并：索引有效性 ----
    def test_case_15(self):
        """mesh_merging: 合并后所有索引指向有效顶点，三角形总数不变。"""
        v1, i1 = grid_mesh(4)
        v2, i2 = grid_mesh(5)
        meshes = [
            {"material": "stone", "vertices": v1, "indices": i1},
            {"material": "stone",
             "vertices": [(p[0] + 50, p[1] + 50, p[2]) for p in v2],
             "indices": i2},
        ]
        result = self.system.mesh_merging(meshes)
        assert len(result) == 1
        merged = result[0]
        assert len(merged["indices"]) == len(i1) + len(i2)
        assert all(0 <= i < len(merged["vertices"]) for i in merged["indices"])
        assert merged["indices"][:len(i1)] == i1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
