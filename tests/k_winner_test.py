#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.keras.utils.generic_utils import custom_object_scope
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

from nupic.tensorflow.constraints import SparseWeights
from nupic.tensorflow.layers import KWinners, KWinners2d
from nupic.tensorflow.layers.k_winners import compute_kwinners

SEED = 18
CUSTOM_OBJECTS = {
    "KWinners": KWinners,
    "KWinners2d": KWinners2d,
    "SparseWeights": SparseWeights,
}


class KWinnersTestBase(object):
    """
    Base module for setting up tests.
    """

    def setUp(self):
        super(KWinnersTestBase, self).setUp()

        # Batch size 2
        x = np.random.random((2, 7)).astype(np.float32) / 2.0
        x[0, 1] = 1.20
        x[0, 2] = 1.10
        x[0, 3] = 1.30
        x[0, 5] = 1.50
        x[1, 0] = 1.11
        x[1, 2] = 1.21
        x[1, 4] = 1.31
        x[1, 6] = 1.22
        self.x1 = x

        # All equal duty cycle for x.
        self.duty_cycles1 = np.full(shape=(7,), fill_value=1.0 / 3.0, dtype=np.float32)

        # Batch size 2
        x2 = np.random.random((2, 6)).astype(np.float32) / 2.0
        x2[0, 0] = 1.50
        x2[0, 1] = 1.02
        x2[0, 2] = 1.10
        x2[0, 3] = 1.30
        x2[0, 5] = 1.03
        x2[1, 0] = 1.11
        x2[1, 1] = 1.04
        x2[1, 2] = 1.20
        x2[1, 3] = 1.60
        x2[1, 4] = 1.01
        x2[1, 5] = 1.05
        self.x2 = x2

        # Unequal duty cycle for x2.
        duty_cycle2 = np.zeros(6, dtype=np.float32)
        duty_cycle2[0] = 1.0 / 2.0
        duty_cycle2[1] = 1.0 / 4.0
        duty_cycle2[2] = 1.0 / 2.0
        duty_cycle2[3] = 1.0 / 4.0
        duty_cycle2[4] = 1.0 / 2.0
        duty_cycle2[5] = 1.0 / 4.0
        self.duty_cycles2 = duty_cycle2

        # Batch size 2, but with negative entries.
        x3 = np.random.random((2, 6)).astype(np.float32) - 0.5
        x3[0, 1] = -1.20
        x3[0, 2] = 1.20
        x3[0, 3] = 1.03
        x3[0, 5] = 1.01
        x3[1, 1] = 1.21
        x3[1, 2] = -1.21
        x3[1, 5] = 1.02
        self.x3 = x3

        # Unequal duty cycle for x3.
        duty_cycle3 = np.zeros(6, dtype=np.float32)
        duty_cycle3[1] = 0.001
        duty_cycle3[2] = 100
        self.duty_cycles3 = duty_cycle3

        # Batch size 1.
        x4 = np.random.random((1, 10)).astype(np.float32) / 2.0
        x4[0, 2] = 1.20
        x4[0, 3] = 1.21
        x4[0, 4] = 1.22
        x4[0, 5] = 1.23
        x4[0, 6] = 1.30
        x4[0, 7] = 1.31
        x4[0, 8] = 1.32
        x4[0, 9] = 1.33
        self.x4 = x4

        # All equal duty cycle for x4.
        self.duty_cycles4 = np.ones(10, dtype=np.float32) / 10


@keras_parameterized.run_all_keras_modes
class KWinnersFowardTest(KWinnersTestBase, keras_parameterized.TestCase):
    """
    Module for testing the forward pass, i.e. the compute_kwinners function.
    """

    def test_one(self):
        """boost strength 0 to 10, k=3, batch size 2"""

        # Setup input.
        x = self.x1
        duty_cycles = self.duty_cycles1

        # Test forward pass through the layer.
        expected = np.zeros(x.shape, dtype=np.float32)
        expected[0, 1] = x[0, 1]
        expected[0, 3] = x[0, 3]
        expected[0, 5] = x[0, 5]
        expected[1, 2] = x[1, 2]
        expected[1, 4] = x[1, 4]
        expected[1, 6] = x[1, 6]

        # Loop over floating point boost strengths.
        for b in np.arange(0.0, 10.0, dtype=np.float32):
            # Build layer with varying boost_strength.
            result = compute_kwinners(x, 3, duty_cycles, boost_strength=b)
            self.assertAllEqual(result, expected)

    def test_two(self):
        """
        Unequal duty cycle, boost strength 0 to 10, k = 3, batch size 2.
        """

        # Setup input.
        x = self.x2
        duty_cycles = self.duty_cycles2

        # Test forward with boost strength of 0.
        expected = np.zeros(x.shape, dtype=np.float32)
        expected[0, 0] = x[0, 0]
        expected[0, 2] = x[0, 2]
        expected[0, 3] = x[0, 3]
        expected[1, 0] = x[1, 0]
        expected[1, 2] = x[1, 2]
        expected[1, 3] = x[1, 3]

        result = compute_kwinners(x, 3, duty_cycles, boost_strength=0.0)
        self.assertAllEqual(result, expected)

        # Test forward again with boost strength of 1.
        expected = np.zeros(x.shape, dtype=np.float32)
        expected[0, 0] = x[0, 0]
        expected[0, 5] = x[0, 5]
        expected[0, 3] = x[0, 3]
        expected[1, 1] = x[1, 1]
        expected[1, 3] = x[1, 3]
        expected[1, 5] = x[1, 5]

        result = compute_kwinners(x, 3, duty_cycles, boost_strength=1.0)
        self.assertAllEqual(result, expected)

        # Test forward again with boost strength from 2 to 10. Should give save result
        # given the differing duty cycles.
        expected = np.zeros(x.shape, dtype=np.float32)
        expected[0, 1] = x[0, 1]
        expected[0, 3] = x[0, 3]
        expected[0, 5] = x[0, 5]
        expected[1, 1] = x[1, 1]
        expected[1, 3] = x[1, 3]
        expected[1, 5] = x[1, 5]

        for b in np.arange(2.0, 10.0, dtype=np.float32):
            result = compute_kwinners(x, 3, duty_cycles, boost_strength=b)
            self.assertAllEqual(result, expected)

    def test_three(self):
        """
        Unequal duty cycle, boost factor 0 (and then over a range), k = 3, batch size 2.
        """

        # Setup input.
        x = self.x3
        duty_cycles = self.duty_cycles3

        # Test forward with boost factor of 0.
        expected = np.zeros(x.shape, dtype=np.float32)
        expected[0, 2] = x[0, 2]
        expected[0, 3] = x[0, 3]
        expected[1, 1] = x[1, 1]
        expected[1, 5] = x[1, 5]

        result = compute_kwinners(x, 2, duty_cycles, boost_strength=0.0)
        self.assertAllEqual(result, expected)

        # Test forward again with boost factor from 1 to 10. Should yield the same
        # result as the negative numbers will never be in the top k and the non-one
        # values have very large duty cycles.
        expected = np.zeros(x.shape, dtype=np.float32)
        expected[0, 3] = x[0, 3]
        expected[0, 5] = x[0, 5]
        expected[1, 1] = x[1, 1]
        expected[1, 5] = x[1, 5]

        for b in np.arange(2.0, 10.0, dtype=np.float32):
            result = compute_kwinners(x, 2, duty_cycles, boost_strength=b)
            self.assertAllEqual(result, expected)

    def test_four(self):
        """
        All equal duty cycle, boost factor 0, k = 0,1, and n, batch size 1.
        """

        # Setup input.
        x = self.x4
        duty_cycles = self.duty_cycles4

        # Test forward with boost factor of 1 and k=0.
        expected = np.zeros(x.shape, dtype=np.float32)

        result = compute_kwinners(x, 0, duty_cycles, boost_strength=1.0)
        self.assertAllEqual(result, expected)

        # Test forward with boost factor of 1 and k=1.
        expected[0, -1] = x[0, -1]

        result = compute_kwinners(x, 1, duty_cycles, boost_strength=1.0)
        self.assertAllEqual(result, expected)

        # Test forward with boost factor of 1 and k=1.
        expected = np.copy(x)

        result = compute_kwinners(x, 10, duty_cycles, boost_strength=1.0)
        self.assertAllEqual(result, expected)

    def test_tie_breaking(self):
        """
        Test k-winners with tie-breaking
        """
        x = self.x2
        # Force tie breaking
        x[0, 5] = x[0, 1]

        # Expected with [0, 1] winning the tie-break
        expected1 = np.zeros_like(x)
        expected1[0, 0] = x[0, 0]
        expected1[0, 1] = x[0, 1]
        expected1[0, 3] = x[0, 3]
        expected1[1, 1] = x[1, 1]
        expected1[1, 3] = x[1, 3]
        expected1[1, 5] = x[1, 5]

        # Expected with [0, 5] winning the tie-break
        expected2 = np.zeros_like(x)
        expected2[0, 0] = x[0, 0]
        expected2[0, 3] = x[0, 3]
        expected2[0, 5] = x[0, 5]
        expected2[1, 1] = x[1, 1]
        expected2[1, 3] = x[1, 3]
        expected2[1, 5] = x[1, 5]
        result = compute_kwinners(
            x, k=3, duty_cycles=self.duty_cycles2, boost_strength=1.0
        )
        result = keras.backend.get_value(result)
        self.assertTrue(
            np.array_equal(result, expected1) or np.array_equal(result, expected2)
        )


@keras_parameterized.run_all_keras_modes
class KWinners1DLayerTest(KWinnersTestBase, keras_parameterized.TestCase):
    """
    Module for testing the 1D KWinners layer.
    """

    def test_one(self):
        # Set input, output, and layer params.
        x = self.x2
        expected = np.zeros(x.shape, dtype=np.float32)
        expected[0, 0] = x[0, 0]
        expected[0, 2] = x[0, 2]
        expected[0, 3] = x[0, 3]
        expected[1, 0] = x[1, 0]
        expected[1, 2] = x[1, 2]
        expected[1, 3] = x[1, 3]
        kwargs = {
            "percent_on": 0.333,
            "k_inference_factor": 1.5,
            "boost_strength": 1.0,
            "boost_strength_factor": 0.5,
            "duty_cycle_period": 1000,
        }

        # Use testing utils to validate layer functionality.
        with custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(
                KWinners, kwargs=kwargs, input_data=x, expected_output=expected
            )

    def test_two(self):
        # Set input, output, and layer params.
        x = self.x2
        expected = np.zeros(x.shape, dtype=np.float32)
        expected[0, 0] = x[0, 0]
        expected[0, 2] = x[0, 2]
        expected[0, 3] = x[0, 3]
        expected[1, 0] = x[1, 0]
        expected[1, 2] = x[1, 2]
        expected[1, 3] = x[1, 3]

        # Test layer within Sequential model.
        # Compile model. Results should be independent on the loss and optimizer.
        model = keras.models.Sequential()
        kw = KWinners(
            percent_on=0.333,
            k_inference_factor=1.5,
            boost_strength=1.0,
            boost_strength_factor=0.5,
            duty_cycle_period=1000,
        )
        model.add(kw)
        model.compile(
            loss="mse",
            optimizer=gradient_descent.GradientDescentOptimizer(0.01),
            run_eagerly=testing_utils.should_run_eagerly(),
        )

        # Ensure there are zero trainable parameters.
        layer = model.layers[0]
        trainable_weights = layer.trainable_weights
        self.assertEmpty(trainable_weights)

        # Validate model prediction (i.e. a forward pass in testing mode).
        result = model.predict_on_batch(x)
        self.assertAllEqual(result, expected)

        # Ensure result doesn't change when only in testing mode.
        result = model.predict_on_batch(x)
        self.assertAllEqual(result, expected)

        # Validate one forward pass in training mode.
        y = np.zeros(x.shape, dtype=np.float32)
        # Expect 2 winners per batch (33% of 6)
        y[0, 0] = x[0, 0]
        y[0, 3] = x[0, 3]
        y[1, 2] = x[1, 2]
        y[1, 3] = x[1, 3]

        loss = model.train_on_batch(x, y)
        self.assertEqual(loss, 0.0)

        # Test values of updated duty cycle.
        old_duty = layer.duty_cycles
        old_duty = old_duty.numpy() if tf.executing_eagerly() else old_duty.eval()
        new_duty = np.array([1.0, 0, 1.0, 2.0, 0, 0], dtype=np.float32) / 2.0
        self.assertAllEqual(old_duty, new_duty)

        # Test forward with updated duty cycle.
        y = np.zeros(x.shape, dtype=np.float32)
        y[0, 1] = x[0, 1]
        y[0, 5] = x[0, 5]
        y[1, 1] = x[1, 1]
        y[1, 5] = x[1, 5]

        loss = model.train_on_batch(x, y)
        self.assertEqual(loss, 0.0)

    def test_three(self):
        """
        Test a series of calls on the layer in training mode.
        """

        x = self.x2
        y = x.copy()

        # Test layer within Sequential model.
        # Compile model. Results should be independent on the loss and optimizer.
        model = keras.models.Sequential()
        kw = KWinners(
            percent_on=0.333,
            k_inference_factor=1.5,
            boost_strength=1.0,
            boost_strength_factor=0.5,
            duty_cycle_period=1000,
        )
        model.add(kw)
        model.compile(
            loss="mse",
            optimizer=gradient_descent.GradientDescentOptimizer(0.01),
            run_eagerly=testing_utils.should_run_eagerly(),
        )

        # Ensure there are zero trainable parameters.
        layer = model.layers[0]
        trainable_weights = layer.trainable_weights
        self.assertEmpty(trainable_weights)

        # "Train" on a sequence of batches. This will only effect the duty cycle.
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)

        # expected = np.zeros_like(x)
        # expected[0, 0] = x[0, 0]
        # expected[0, 5] = x[0, 5]
        # expected[1, 2] = x[1, 2]
        # expected[1, 3] = x[1, 3]
        # result = model(x, training=True)
        # self.assertAllEqual(result, expected)

        # Validate model prediction (i.e. a forward pass in testing mode).
        result = model.predict_on_batch(x)
        expected = np.zeros_like(x)
        expected[0, 0] = x[0, 0]
        expected[0, 1] = x[0, 1]
        expected[0, 5] = x[0, 5]
        expected[1, 2] = x[1, 2]
        expected[1, 3] = x[1, 3]
        expected[1, 4] = x[1, 4]
        self.assertAllEqual(result, expected)


@keras_parameterized.run_all_keras_modes
class KWinners2DLayerTest(keras_parameterized.TestCase):
    """
    Module for testing the 1D KWinners layer.
    """

    def setUp(self):
        super(KWinners2DLayerTest, self).setUp()

        # Batch size 1
        x = np.random.random((1, 3, 2, 2)).astype(np.float32) / 2.0
        x[0, 0, 1, 0] = 1.10
        x[0, 0, 1, 1] = 1.20
        x[0, 1, 0, 1] = 1.21
        x[0, 2, 1, 0] = 1.30
        self.x1 = x

        # Batch size 2
        x = np.random.random((2, 3, 2, 2)).astype(np.float32) / 2.0

        x[0, 0, 1, 0] = 1.10
        x[0, 0, 1, 1] = 1.20
        x[0, 1, 0, 1] = 1.21
        x[0, 2, 1, 0] = 1.30

        x[1, 0, 0, 0] = 1.40
        x[1, 1, 0, 0] = 1.50
        x[1, 1, 0, 1] = 1.60
        x[1, 2, 1, 1] = 1.70
        self.x2 = x

    def test_one(self):
        """Equal duty cycle, boost strength 0, k=4, batch size 1."""
        x = self.x1
        expected = np.zeros_like(x)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]

        n = np.prod(x.shape[1:])
        kwargs = {
            "percent_on": 4 / n,
            "k_inference_factor": 1.0,
            "boost_strength": 0.0,
            "data_format": "channels_first",
        }
        with custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(
                KWinners2d, kwargs=kwargs, input_data=x, expected_output=expected
            )

    def test_two(self):
        """Equal duty cycle, boost strength 0, k=3."""
        x = self.x1
        expected = np.zeros(x.shape)
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        n = np.prod(x.shape[1:])
        kwargs = {
            "percent_on": 3 / n,
            "k_inference_factor": 1.0,
            "boost_strength": 0.0,
            "data_format": "channels_first",
        }

        with custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(
                KWinners2d, kwargs=kwargs, input_data=x, expected_output=expected
            )

    def test_three(self):
        """Equal duty cycle, boost strength=0, k=4, batch size=2."""
        x = self.x2
        expected = np.zeros(x.shape)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 0, 0, 0] = x[1, 0, 0, 0]
        expected[1, 1, 0, 0] = x[1, 1, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        n = np.prod(x.shape[1:])
        kwargs = {
            "percent_on": 4 / n,
            "k_inference_factor": 1.0,
            "boost_strength": 0.0,
            "data_format": "channels_first",
        }
        with custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(
                KWinners2d, kwargs=kwargs, input_data=x, expected_output=expected
            )

    def test_four(self):
        """Equal duty cycle, boost strength=0, k=3, batch size=2."""
        x = self.x2
        expected = np.zeros(x.shape)
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 1, 0, 0] = x[1, 1, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        n = np.prod(x.shape[1:])
        kwargs = {
            "percent_on": 3 / n,
            "k_inference_factor": 1.0,
            "boost_strength": 0.0,
            "data_format": "channels_first",
        }
        with custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(
                KWinners2d, kwargs=kwargs, input_data=x, expected_output=expected
            )

    def test_five(self):
        """
        Test a series of calls on the layer in training mode.
        """

        x = self.x2
        y = x.copy()

        kwargs = {
            "percent_on": 0.25,
            "k_inference_factor": 0.5,
            "boost_strength": 1.0,
            "boost_strength_factor": 0.5,
            "duty_cycle_period": 1000,
            "data_format": "channels_first",
        }

        # Test layer within Sequential model.
        # Compile model. Results should be independent on the loss and optimizer.
        model = keras.models.Sequential()
        kw = KWinners2d(**kwargs)
        model.add(kw)
        model.compile(
            loss="mse",
            optimizer=gradient_descent.GradientDescentOptimizer(0.01),
            run_eagerly=testing_utils.should_run_eagerly(),
        )

        # Ensure there are zero trainable parameters.
        layer = model.layers[0]
        trainable_weights = layer.trainable_weights
        self.assertEmpty(trainable_weights)

        # "Train" on a sequence of batches. This will only effect the duty cycle.
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)

        # Validate model prediction (i.e. a forward pass in testing mode).
        expected = np.zeros_like(x)
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]
        result = model.predict_on_batch(x)
        self.assertAllEqual(result, expected)


if __name__ == "__main__":
    test.main()
