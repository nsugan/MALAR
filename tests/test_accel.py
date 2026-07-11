from malar.core.accel import device_report, parallel_map


def test_parallel_map_preserves_order_and_edges():
    assert parallel_map(lambda x: x * x, [1, 2, 3, 4, 5]) == [1, 4, 9, 16, 25]
    assert parallel_map(lambda x: x, []) == []
    assert parallel_map(lambda x: x + 1, [10]) == [11]


def test_device_report_shape_is_honest():
    r = device_report()
    for k in ("cpu_count", "parallel_workers", "gpu_accelerated", "cpu_only", "not_wired"):
        assert k in r
    assert isinstance(r["gpu_accelerated"], list) and r["gpu_accelerated"]
    assert r["parallel_workers"] >= 2
    # NPU / iGPU must be reported as NOT wired (no overselling)
    assert any("NPU" in s for s in r["not_wired"])
