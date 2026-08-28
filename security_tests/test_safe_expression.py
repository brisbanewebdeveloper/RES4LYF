import unittest

import torch

from safe_expression import MathExpression


class MathExpressionTests(unittest.TestCase):
    def test_evaluates_tensor_arithmetic_and_allowed_functions(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        expression = MathExpression("torch.sqrt(s ** 2) + x", {"s", "x"})

        result = expression.evaluate({"s": values, "x": 2.0})

        self.assertTrue(torch.equal(result, torch.tensor([3.0, 4.0, 5.0])))

    def test_evaluates_allowed_module_constants(self):
        expression = MathExpression("s * np.pi", {"s"})

        result = expression.evaluate({"s": torch.tensor([1.0])})

        self.assertTrue(torch.allclose(result, torch.tensor([torch.pi])))

    def test_evaluates_legacy_sampler_formulas(self):
        cases = [
            ("1 / (sigma.exp() + 1)", "sigma", torch.tensor([0.0, 1.0]), torch.tensor([0.5, 0.26894143])),
            ("((1 - t) / t).log()", "t", torch.tensor([0.25, 0.5]), torch.tensor([1.0986123, 0.0])),
        ]

        for formula, variable, value, expected in cases:
            with self.subTest(formula=formula):
                expression = MathExpression(formula, {variable})
                result = expression.evaluate({variable: value})
                self.assertTrue(torch.allclose(result, expected))

    def test_rejects_python_execution_primitives(self):
        formulas = [
            "__import__('urllib.request').urlopen('https://example.com')",
            "torch.load('payload.pt')",
            "s.__class__",
            "(lambda: 1)()",
            "[value for value in s]",
        ]

        for formula in formulas:
            with self.subTest(formula=formula):
                expression = MathExpression(formula, {"s"})
                with self.assertRaises(ValueError):
                    expression.evaluate({"s": torch.tensor([1.0])})

    def test_rejects_excessive_power_exponents(self):
        expression = MathExpression("2 ** (2 ** 20)", set())

        with self.assertRaisesRegex(ValueError, "Power exponent is too large"):
            expression.evaluate({})


if __name__ == "__main__":
    unittest.main()
