import base64
import pickle
import unittest

import torch

from safe_serialization import deserialize_conditioning, serialize_conditioning


class UnsupportedPayload:
    def __reduce__(self):
        return eval, ("1 + 1",)


class SafeSerializationTests(unittest.TestCase):
    def test_round_trips_conditioning_tensors_and_metadata(self):
        conditioning = [
            [
                torch.tensor([[1.0, 2.0]]),
                {
                    "pooled_output": torch.tensor([3.0]),
                    "start_percent": 0.25,
                },
            ]
        ]

        restored = deserialize_conditioning(serialize_conditioning(conditioning))

        self.assertTrue(torch.equal(restored[0][0], conditioning[0][0]))
        self.assertTrue(torch.equal(restored[0][1]["pooled_output"], conditioning[0][1]["pooled_output"]))
        self.assertEqual(restored[0][1]["start_percent"], 0.25)

    def test_rejects_legacy_pickle_payloads(self):
        payload = base64.b64encode(pickle.dumps({"conditioning": []})).decode("ascii")

        with self.assertRaisesRegex(ValueError, "legacy pickle payloads are not accepted"):
            deserialize_conditioning(payload)

    def test_rejects_unsupported_objects_before_serialization(self):
        with self.assertRaisesRegex(TypeError, "unsupported type"):
            serialize_conditioning(UnsupportedPayload())

    def test_rejects_versioned_pickle_data_without_loading_it(self):
        encoded = base64.b64encode(pickle.dumps(UnsupportedPayload())).decode("ascii")

        with self.assertRaisesRegex(ValueError, "invalid format"):
            deserialize_conditioning("res4lyf-conditioning-v2:" + encoded)

    def test_rejects_invalid_base64(self):
        with self.assertRaisesRegex(ValueError, "not valid base64"):
            deserialize_conditioning("res4lyf-conditioning-v2:not-base64!")


if __name__ == "__main__":
    unittest.main()
