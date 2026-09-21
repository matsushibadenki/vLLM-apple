import unittest

from vllm_apple.generative_chunk_scheduler import build_generative_chunk_plan


class GenerativeChunkSchedulerTests(unittest.TestCase):
    def test_temporal_overlap_spatial_tiles_and_dependencies(self):
        plan = build_generative_chunk_plan(
            frames=9, height=6, width=8, temporal_chunk=4, temporal_overlap=1,
            tile_height=4, tile_width=4, bytes_per_attention_element=2,
            attention_channels=8, maximum_concurrency=4,
            memory_ceiling_bytes=4096,
        )
        self.assertEqual(len(plan.tasks), 12)
        first_tile = [task for task in plan.tasks if task.region.y == 0 and task.region.x == 0]
        self.assertEqual([task.temporal_context_count for task in first_tile], [4, 5, 2])
        self.assertEqual(first_tile[0].dependencies, ())
        self.assertEqual(first_tile[1].dependencies, (first_tile[0].task_id,))
        self.assertLessEqual(plan.estimated_peak_bytes, 4096)

    def test_memory_ceiling_clamps_concurrency_deterministically(self):
        arguments = dict(
            frames=8, height=4, width=4, temporal_chunk=4, temporal_overlap=0,
            tile_height=4, tile_width=4, bytes_per_attention_element=2,
            attention_channels=8, maximum_concurrency=8,
            memory_ceiling_bytes=2048,
        )
        first = build_generative_chunk_plan(**arguments)
        second = build_generative_chunk_plan(**arguments)
        self.assertEqual(first, second)
        self.assertEqual(first.maximum_concurrency, 2)

    def test_invalid_overlap_and_single_state_overflow_fail_closed(self):
        base = dict(
            frames=8, height=4, width=4, temporal_chunk=4, temporal_overlap=0,
            tile_height=4, tile_width=4, bytes_per_attention_element=2,
            attention_channels=8, maximum_concurrency=1,
            memory_ceiling_bytes=1024,
        )
        with self.assertRaises(ValueError):
            build_generative_chunk_plan(**(base | {"temporal_overlap": 4}))
        with self.assertRaisesRegex(ValueError, "single attention"):
            build_generative_chunk_plan(**(base | {"memory_ceiling_bytes": 100}))


if __name__ == "__main__":
    unittest.main()
