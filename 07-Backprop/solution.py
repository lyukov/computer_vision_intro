from interface import *


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values,
                n - batch size, ... - arbitrary input shape
        :return: np.array((n, ...)), output values,
                n - batch size, ... - arbitrary output shape (same as input)
        """
        # your code here \/
        return inputs.clip(0)
        # your code here /\

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs,
                n - batch size, ... - arbitrary output shape
        :return: np.array((n, ...)), dLoss/dInputs,
                n - batch size, ... - arbitrary input shape (same as output)
        """
        # your code here \/
        inputs = self.forward_inputs
        return grad_outputs * (inputs > 0)
        # your code here /\


# ============================== 2.1.2 Softmax ===============================
class Softmax(Layer):
    def forward(self, inputs):
        """
        :param inputs: np.array((n, d)), input values,
                n - batch size, d - number of units
        :return: np.array((n, d)), output values,
                n - batch size, d - number of units
        """
        # your code here \/
        tmp = np.exp(inputs).T
        return (tmp / tmp.sum(axis=0)).T
        # your code here /\

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs,
                n - batch size, d - number of units
        :return: np.array((n, d)), dLoss/dInputs,
                n - batch size, d - number of units
        """
        # your code here \/
        outputs = self.forward_outputs
        n, d = outputs.shape
        tmp = -np.matmul(outputs.reshape((n, d, 1)), outputs.reshape((n, 1, d)))
        tmp += np.eye(d) * outputs.reshape((n, d, 1))
        return np.matmul(grad_outputs.reshape((n, 1, d)), tmp).reshape((n, d))
        # your code here /\


# =============================== 2.1.3 Dense ================================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_shape = (units,)
        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units, = self.output_shape

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

    def forward(self, inputs):
        """
        :param inputs: np.array((n, d)), input values,
                n - batch size, d - number of input units
        :return: np.array((n, c)), output values,
                n - batch size, c - number of output units
        """
        # your code here \/
        batch_size, input_units = inputs.shape
        output_units, = self.output_shape
        return np.matmul(inputs, self.weights) + self.biases
        # your code here /\

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs,
                n - batch size, c - number of output units
        :return: np.array((n, d)), dLoss/dInputs,
                n - batch size, d - number of input units
        """
        # your code here \/
        batch_size, output_units = grad_outputs.shape
        input_units, = self.input_shape
        inputs = self.forward_inputs

        # Don't forget to update current gradients:
        # dLoss/dWeights
        self.weights_grad = np.matmul(
            grad_outputs.reshape(batch_size, output_units, 1),
            inputs.reshape(batch_size, 1, input_units)
        ).mean(axis=0).T
        # dLoss/dBiases
        self.biases_grad = grad_outputs.mean(axis=0)
        return np.matmul(grad_outputs, self.weights.T)
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def __call__(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n,)), loss scalars for batch
        """
        # your code here \/
        batch_size, output_units = y_gt.shape
        return -(y_gt * np.log(1e-10 + y_pred)).sum(axis=1)
        # your code here /\

    def gradient(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n, d)), gradient loss to y_pred
        """
        # your code here \/
        return - y_gt / (1e-10 + y_pred)
        # your code here /\


# ================================ 2.3.1 SGD =================================
class SGD(Optimizer):
    def __init__(self, lr):
        self._lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            # your code here \/
            assert parameter_shape == parameter.shape
            assert parameter_shape == parameter_grad.shape
            return parameter - self._lr * parameter_grad
            # your code here /\

        return updater


# ============================ 2.3.2 SGDMomentum =============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self._lr = lr
        self._momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            # your code here \/
            assert parameter_shape == parameter.shape
            assert parameter_shape == parameter_grad.shape
            assert parameter_shape == updater.inertia.shape

            # Don't forget to update the current inertia tensor:
            updater.inertia = self._momentum * updater.inertia + self._lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ======================= 2.4 Train and test on MNIST ========================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.001, 0.5))
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(32, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())
    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=30, epochs=2, x_valid=x_valid, y_valid=y_valid)
    # your code here /\
    return model

# ============================================================================
