"""
游戏内网格压缩与几何优化系统 - 验收测试
运行方式：python -m pytest tests/test_mesh_compression.py -q
共 15 个测试用例
"""
import pytest
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.mesh_compression import MeshCompression


def make_grid(size=6):
    """生成 size x size 的平面网格（三角形顺序随机打乱）。"""
    vertices = [(float(x), float(y), 0.0)
                for y in range(size + 1) for x in range(size + 1)]

    def vid(x, y):
        return y * (size + 1) + x

    triangles = []
    for y in range(size):
        for x in range(size):
            triangles.append((vid(x, y), vid(x + 1, y), vid(x, y + 1)))
            triangles.append((vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)))
    rng = random.Random(42)
    rng.shuffle(triangles)
    indices = [v for tri in triangles for v in tri]
    return vertices, indices


def make_depth_triangles():
    """生成屏幕空间重叠、深度递增的三角形（初始为后往前的不良顺序）。"""
    vertices = []
    indices = []
    for i in range(6):
        base = len(vertices)
        z = float(i)
        vertices += [(-1.0, -1.0, z), (1.0, -1.0, z), (0.0, 1.0, z)]
        indices += [base, base + 1, base + 2]
    # 故意按从远到近排序（深度大的先画）
    order = sorted(range(6), key=lambda i: -vertices[i * 3][2])
    shuffled = []
    for i in order:
        shuffled += indices[i * 3:i * 3 + 3]
    return vertices, shuffled


class TestMeshCompression:
    def setup_method(self):
        self.system = MeshCompression()

    # ---- 网格压缩：顶点量化 + 索引压缩 ----

    def test_case_01(self):
        """顶点量化：量化后的坐标为指定位数范围内的整数"""
        vertices, indices = make_grid()
        bits = 12
        result = self.system.compress_mesh(vertices, indices, quantization_bits=bits)
        quantized = result["quantized_vertices"]
        assert len(quantized) == len(vertices)
        max_val = (1 << bits) - 1
        for q in quantized:
            for component in q:
                assert isinstance(component, int)
                assert 0 <= component <= max_val

    def test_case_02(self):
        """索引压缩：压缩后的索引缓冲比原始 32 位索引更小且可无损解码"""
        vertices, indices = make_grid()
        result = self.system.compress_mesh(vertices, indices)
        compressed = result["compressed_indices"]
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(indices) * 4
        decoded = self.system.decompress_mesh(result)
        assert decoded["indices"] == indices

    def test_case_03(self):
        """网格压缩：整体体积变小，且解压后的顶点位置在量化误差内"""
        vertices, indices = make_grid()
        result = self.system.compress_mesh(vertices, indices)
        assert result["compressed_size"] < result["original_size"]
        assert result["compression_ratio"] > 1.0
        decoded = self.system.decompress_mesh(result)
        extent = max(result["bbox_max"][a] - result["bbox_min"][a] for a in range(3))
        tolerance = extent / ((1 << result["quantization_bits"]) - 1) * 1.01
        for original, restored in zip(vertices, decoded["vertices"]):
            for a in range(3):
                assert abs(original[a] - restored[a]) <= tolerance

    # ---- 几何优化：顶点缓存优化 + 过度绘制优化 ----

    def test_case_04(self):
        """顶点缓存优化：优化后的缓存命中率不低于优化前"""
        vertices, indices = make_grid()
        result = self.system.geometry_optimization(vertices, indices, cache_size=16)
        assert result["cache_hit_rate_after"] >= result["cache_hit_rate_before"]
        assert result["cache_hit_rate_after"] > 0.0

    def test_case_05(self):
        """过度绘制优化：优化后的过度绘制估算值减少"""
        vertices, indices = make_depth_triangles()
        result = self.system.geometry_optimization(vertices, indices)
        assert result["overdraw_before"] > 0
        assert result["overdraw_after"] < result["overdraw_before"]

    def test_case_06(self):
        """几何优化：不改变网格本身（三角形集合与顶点保持不变）"""
        vertices, indices = make_grid()
        result = self.system.geometry_optimization(vertices, indices)
        before = sorted(tuple(sorted(indices[i:i + 3]))
                        for i in range(0, len(indices), 3))
        after = sorted(tuple(sorted(result["indices"][i:i + 3]))
                       for i in range(0, len(result["indices"]), 3))
        assert before == after
        assert result["vertices"] == [tuple(v) for v in vertices]

    # ---- LOD 生成：误差度量 + 自动生成 ----

    def test_case_07(self):
        """自动生成：一次调用生成请求的多个 LOD 级别"""
        vertices, indices = make_grid()
        levels = self.system.lod_generation(vertices, indices, num_levels=4)
        assert len(levels) == 4
        assert [lvl["level"] for lvl in levels] == [0, 1, 2, 3]

    def test_case_08(self):
        """误差度量：每个级别都带误差，LOD0 误差为 0，级别越高累计误差越大"""
        vertices, indices = make_grid()
        levels = self.system.lod_generation(vertices, indices, num_levels=3)
        assert levels[0]["error"] == 0.0
        assert levels[0]["cumulative_error"] == 0.0
        for lvl in levels[1:]:
            assert lvl["error"] >= 0.0
            assert lvl["cumulative_error"] >= lvl["error"]
        assert levels[-1]["cumulative_error"] > 0.0

    def test_case_09(self):
        """LOD 生成：级别越高三角形越少"""
        vertices, indices = make_grid()
        levels = self.system.lod_generation(vertices, indices, num_levels=3,
                                            reduction_ratio=0.5)
        counts = [lvl["triangle_count"] for lvl in levels]
        assert counts[0] == len(indices) // 3
        assert counts[1] < counts[0]
        assert counts[2] < counts[1]

    # ---- 网格简化：边折叠 + 属性保持 ----

    def test_case_10(self):
        """边折叠：简化通过边折叠实现，三角形数量降到目标附近"""
        vertices, indices = make_grid()
        result = self.system.mesh_simplification(vertices, indices, target_ratio=0.5)
        assert result["collapsed_edges"] > 0
        target = max(1, round(len(indices) // 3 * 0.5))
        assert result["triangle_count"] <= target
        assert len(result["indices"]) == result["triangle_count"] * 3

    def test_case_11(self):
        """网格简化：输出索引合法且顶点数组被压缩"""
        vertices, indices = make_grid()
        result = self.system.mesh_simplification(vertices, indices, target_ratio=0.5)
        assert len(result["vertices"]) < len(vertices)
        assert result["indices"]
        assert max(result["indices"]) < len(result["vertices"])
        assert min(result["indices"]) >= 0

    def test_case_12(self):
        """属性保持：简化后属性数量与顶点一致，且插值结果在原属性范围内"""
        vertices, indices = make_grid()
        uvs = [(v[0] / 6.0, v[1] / 6.0) for v in vertices]
        colors = [(v[0] / 6.0, 0.5, v[1] / 6.0) for v in vertices]
        result = self.system.mesh_simplification(
            vertices, indices, target_ratio=0.5,
            attributes={"uv": uvs, "color": colors},
        )
        attrs = result["attributes"]
        assert len(attrs["uv"]) == len(result["vertices"])
        assert len(attrs["color"]) == len(result["vertices"])
        for uv in attrs["uv"]:
            assert 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0
        for color in attrs["color"]:
            assert all(0.0 <= c <= 1.0 for c in color)

    # ---- 网格合并：材质合并 + 顶点合并 ----

    def test_case_13(self):
        """网格合并：多个网格合并为一个网格，三角形总数不变"""
        v1, i1 = make_grid(2)
        v2 = [(x + 10.0, y, z) for x, y, z in v1]
        meshes = [
            {"vertices": v1, "indices": i1, "material": "stone"},
            {"vertices": v2, "indices": i1, "material": "wood"},
        ]
        result = self.system.mesh_merging(meshes)
        assert result["triangle_count"] == 2 * (len(i1) // 3)
        assert max(result["indices"]) < len(result["vertices"])

    def test_case_14(self):
        """材质合并：相同材质的网格合并到同一个子网格"""
        v, i = make_grid(2)
        meshes = [
            {"vertices": v, "indices": i, "material": "stone"},
            {"vertices": [(x + 5.0, y, z) for x, y, z in v], "indices": i, "material": "wood"},
            {"vertices": [(x + 20.0, y, z) for x, y, z in v], "indices": i, "material": "stone"},
        ]
        result = self.system.mesh_merging(meshes)
        assert len(result["submeshes"]) == 2
        materials = [s["material"] for s in result["submeshes"]]
        assert materials == ["stone", "wood"]
        stone = result["submeshes"][0]
        assert stone["index_count"] == 2 * len(i)

    def test_case_15(self):
        """顶点合并：重合的顶点被去重，索引被正确重映射"""
        v, i = make_grid(2)
        meshes = [
            {"vertices": v, "indices": i, "material": "stone"},
            {"vertices": list(v), "indices": list(i), "material": "stone"},
        ]
        result = self.system.mesh_merging(meshes)
        assert result["original_vertex_count"] == 2 * len(v)
        assert result["merged_vertex_count"] == len(v)
        # 两个网格共享同一批顶点，合并后索引值域不超过去重后的顶点数
        assert max(result["indices"]) < len(v)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
