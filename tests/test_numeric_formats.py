import unittest
from dataclasses import replace

from vllm_apple.numeric_formats import (
    ConversionAdapter, ConversionRegistry, DEFAULT_CONVERSION_REGISTRY,
    TensorGeometry,
    NumericFormatDescriptor, conversion_plan, convert_nvfp4_to_int8, decode_nvfp4,
)


class NumericFormatTests(unittest.TestCase):
    def test_multidimensional_conversion_and_content_binding(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 34)
        packed = bytes([0x22] * 17)
        scales = bytes([56, 64, 72, 80])  # 1, 2, 4, 8
        for geometry, expected in (
            (TensorGeometry((2, 17), 1), (1,) * 16 + (2,) + (4,) * 16 + (8,)),
            (TensorGeometry((17, 2), 0), (1, 2) * 16 + (4, 8)),
        ):
            converted = convert_nvfp4_to_int8(descriptor, packed, scales, 1, geometry=geometry)
            self.assertEqual(converted.reference_values(), expected)
            self.assertEqual(decode_nvfp4(descriptor, packed, scales, 1, geometry=geometry), expected)
            converted.verify_source(descriptor, packed, scales, 1, geometry=geometry)
            with self.assertRaises(ValueError):
                converted.verify_source(descriptor, packed, scales, 1)
            with self.assertRaises(ValueError):
                replace(converted, block_scales=bytes([56] * 4))
        first = convert_nvfp4_to_int8(descriptor, packed, scales, 1,
                                     geometry=TensorGeometry((2, 17), 1))
        other = TensorGeometry((17, 2), 0)
        with self.assertRaises(ValueError):
            replace(first, plan=replace(first.plan, geometry=other))
        with self.assertRaises(ValueError):
            first.verify_source(descriptor, packed, scales, 1, geometry=other)

    def test_geometry_execution_rejects_invalid_scales_and_padding(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 3)
        geometry = TensorGeometry((3, 1), 1)
        for packed, scales, global_scale in (
            (bytes([0x22, 0x22]), bytes([56] * 3), 1),
            (bytes([0x22, 2]), bytes([56]), 1),
            (bytes([0x22, 2]), bytes([56, 127, 56]), 1),
            (bytes([0x22, 2]), bytes([56] * 3), float("inf")),
        ):
            with self.assertRaises(ValueError):
                convert_nvfp4_to_int8(descriptor, packed, scales, global_scale, geometry=geometry)
        result = convert_nvfp4_to_int8(descriptor, bytes([0x22, 2]), bytes([56, 64, 72]), 1,
                                      geometry=geometry)
        self.assertEqual(result.reference_values(), (1, 2, 4))

    def test_geometry_scale_mapping_restarts_at_axis_boundaries(self):
        rows = TensorGeometry((2, 17), scale_axis=1)
        self.assertEqual(rows.scale_shape, (2, 2))
        self.assertEqual(rows.scale_count, 4)
        self.assertEqual([rows.scale_index(i) for i in range(34)],
                         [0] * 16 + [1] + [2] * 16 + [3])
        columns = TensorGeometry((17, 2), scale_axis=0)
        self.assertEqual(columns.scale_shape, (2, 2))
        self.assertEqual([columns.scale_index(i) for i in range(34)],
                         [0, 1] * 16 + [2, 3])
        volume = TensorGeometry((2, 3, 4), scale_axis=1, block_size=2)
        self.assertEqual(volume.scale_shape, (2, 2, 4))
        self.assertEqual([volume.scale_index(i) for i in range(24)],
                         list(range(4)) * 2 + list(range(4, 8))
                         + list(range(8, 12)) * 2 + list(range(12, 16)))

    def test_tensor_plan_binds_geometry_without_changing_legacy_plan(self):
        source = NumericFormatDescriptor("nvfp4_e2m1", 32)
        registry = DEFAULT_CONVERSION_REGISTRY
        first = registry.plan_tensor(source, TensorGeometry((2, 16), 1))
        second = registry.plan_tensor(source, TensorGeometry((16, 2), 0))
        self.assertEqual(first.format_plan, conversion_plan(source))
        self.assertNotEqual(first.plan_id, second.plan_id)
        self.assertEqual(first.plan_id, registry.plan_tensor(source, first.geometry).plan_id)
        for geometry in (TensorGeometry((16,), 0), TensorGeometry((32,), 0, 8)):
            with self.assertRaises(ValueError):
                registry.plan_tensor(source, geometry)

    def test_invalid_geometry_is_rejected(self):
        for shape, axis, block in (((), 0, 16), ([16], 0, 16), ((True,), 0, 16),
                                   ((0,), 0, 16), ((65537,), 0, 16),
                                   ((1,) * 9, 0, 16), ((16,), -1, 16),
                                   ((16,), 1, 16), ((16,), True, 16),
                                   ((16,), 0, 0), ((16,), 0, True)):
            with self.subTest(shape=shape, axis=axis, block=block), self.assertRaises(ValueError):
                TensorGeometry(shape, axis, block)
        geometry = TensorGeometry((16,), 0)
        for index in (-1, 16, True, 0.5):
            with self.assertRaises(ValueError):
                geometry.scale_index(index)

    def test_registry_is_immutable_and_requires_unambiguous_route(self):
        original = DEFAULT_CONVERSION_REGISTRY
        first = original.adapters[0]
        alternate = replace(first, adapter_id="alternate_cpu_v1")
        registry = original.register(alternate)
        source = replace(first.source, elements=16)
        self.assertEqual(len(original.adapters), 1)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            registry.plan(source)
        plan = registry.plan(source, adapter_id=alternate.adapter_id)
        self.assertEqual(plan.adapter, alternate.adapter_id)
        self.assertEqual(plan.target.elements, 16)
        self.assertNotEqual(plan.plan_id, conversion_plan(source).plan_id)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.register(first)
        with self.assertRaises(ValueError):
            ConversionRegistry([first])
        with self.assertRaises(ValueError):
            ConversionAdapter("bad-id", first.source, first.target)
        with self.assertRaises(ValueError):
            replace(first, source=source)

    def test_registry_exact_target_selection_and_fail_closed(self):
        first = DEFAULT_CONVERSION_REGISTRY.adapters[0]
        second = replace(first, adapter_id="other_target_v1",
                         target=replace(first.target, encoding="test_only_format"))
        registry = DEFAULT_CONVERSION_REGISTRY.register(second)
        source = replace(first.source, elements=32)
        target = replace(second.target, elements=32)
        self.assertEqual(registry.plan(source, target=target).adapter, second.adapter_id)
        for kwargs in ({"target": second.target}, {"adapter_id": "missing"},
                       {"target": target, "adapter_id": first.adapter_id}):
            with self.assertRaises(ValueError):
                registry.plan(source, **kwargs)
        with self.assertRaises(ValueError):
            registry.plan(replace(source, layout="swizzled"))
        with self.assertRaises(ValueError):
            ConversionRegistry().plan(source)
        # A registered plan alone must not authorize the built-in executor.
        converted = convert_nvfp4_to_int8(source, bytes(16), bytes([56, 56]), 1)
        with self.assertRaises(ValueError):
            replace(converted, plan=registry.plan(source, target=target))

    def test_content_binding_and_source_verification(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 16)
        packed, scales = bytes([0x22] * 8), bytes([56])
        converted = convert_nvfp4_to_int8(descriptor, packed, scales, 1)
        converted.verify_source(descriptor, packed, scales, 1.0)
        self.assertEqual(converted, convert_nvfp4_to_int8(descriptor, packed, scales, 1.0))
        for payload, scale, global_scale in (
            (bytes(8), scales, 1), (packed, bytes([57]), 1), (packed, scales, 2)
        ):
            with self.assertRaises(ValueError):
                converted.verify_source(descriptor, payload, scale, global_scale)
        for change in (
            {"payload": bytes(16)}, {"payload": bytes(15)},
            {"block_scales": bytes([57])}, {"global_scale": 2},
            {"source_digest": "invalid"},
            {"plan": replace(converted.plan, adapter="unknown")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(converted, **change)
        forged = replace(converted, source_digest="0" * 64)
        with self.assertRaises(ValueError):
            forged.verify_source(descriptor, packed, scales, 1)

    def test_signed_zero_sources_remain_distinguishable(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 2)
        positive = convert_nvfp4_to_int8(descriptor, bytes([0]), bytes([56]), 1)
        negative = convert_nvfp4_to_int8(descriptor, bytes([0x88]), bytes([56]), 1)
        self.assertEqual(positive.target_digest, negative.target_digest)
        self.assertNotEqual(positive.source_digest, negative.source_digest)
        with self.assertRaises(ValueError):
            positive.verify_source(descriptor, bytes([0x88]), bytes([56]), 1)

    def test_all_codes_and_finite_scale_codes_preserve_numeric_values(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 16)
        packed = bytes(range(0x10, 0x100, 0x22))
        expected = (0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6)
        self.assertEqual(decode_nvfp4(descriptor, packed, bytes([56]), 1), expected)
        for scale in range(127):
            with self.subTest(scale=scale):
                converted = convert_nvfp4_to_int8(descriptor, packed, bytes([scale]), .75)
                self.assertEqual(converted.reference_values(), decode_nvfp4(
                    descriptor, packed, bytes([scale]), .75
                ))
                self.assertEqual(len(converted.payload), 16)

    def test_partial_blocks_and_padding(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 17)
        converted = convert_nvfp4_to_int8(descriptor, bytes([0x22] * 8 + [2]), bytes([56, 64]), 1)
        self.assertEqual(converted.reference_values(), (1,) * 16 + (2,))
        with self.assertRaises(ValueError):
            convert_nvfp4_to_int8(descriptor, bytes([0x22] * 9), bytes([56, 64]), 1)

    def test_variant_identity_and_invalid_metadata(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 16)
        self.assertEqual(conversion_plan(descriptor).plan_id, conversion_plan(descriptor).plan_id)
        self.assertNotEqual(conversion_plan(descriptor).plan_id,
                            conversion_plan(replace(descriptor, elements=32)).plan_id)
        for change in ({"layout": "swizzled"}, {"block_size": 32}, {"encoding": "mxfp4"}):
            with self.assertRaises(ValueError):
                conversion_plan(replace(descriptor, **change))
        for scale, global_scale in ((127, 1), (128, 1), (56, float("nan")), (56, -1), (126, 1e308)):
            with self.assertRaises(ValueError):
                convert_nvfp4_to_int8(descriptor, bytes(8), bytes([scale]), global_scale)
        for count in (True, 0, 65537):
            with self.assertRaises(ValueError):
                NumericFormatDescriptor("nvfp4_e2m1", count)


if __name__ == "__main__":
    unittest.main()
