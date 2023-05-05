from pi_theremin import capGain
import numpy as np

def test_cap_gain() -> None:
    # Test capGain with an input that should not change (when amp not > cap)
    input_samples = np.array([1000, 2000, -1000, -2000], dtype=np.int16)
    expected_output = input_samples.copy()
    output = capGain(input_samples)
    np.testing.assert_array_equal(output, expected_output)

    # Test capGain with an input that should be reduced (when amp > capped value)
    input_samples = np.array([32000, -32000], dtype=np.int16)
    expected_output = np.array([16000, -16000], dtype=np.int16)
    output = capGain(input_samples)
    np.testing.assert_array_equal(output, expected_output)