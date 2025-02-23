public class NeuralNetwork {
    int layerCount;
    int neuronsPerHiddenLayer;

    static final int INPUT_NODES = 28*28;
    static final int FINAL_LAYER_NEURONS = 10;

    Matrix[] weights;
    Matrix[] biases;

    static final int BATCH_SIZE = 300;
    Matrix[] weightChanges;
    Matrix[] biasChanges;

    Matrix[] activations;
    Matrix[] lastLayerActivationGradients;

    public NeuralNetwork(int hiddenLayers, int neuronsPerHiddenLayer) {
        if(hiddenLayers <= 0 || neuronsPerHiddenLayer <= 0) {
            throw new IllegalArgumentException("Must have positive hidden layers and neurons per hidden layer");
        }
        this.layerCount = hiddenLayers + 1;
        this.neuronsPerHiddenLayer = neuronsPerHiddenLayer;
        setLastLayerActivationGradients();
        instantiateParameters();
    }

    private void setLastLayerActivationGradients() {
        lastLayerActivationGradients = new Matrix[10];
        for(int currentDigit = 0; currentDigit < 10; currentDigit++) {
            Matrix lastLayerGradient = new Matrix(1, 10);
            for(int col = 0; col < 10; col++) {
                // activation gradient is negative for the correct digit
                lastLayerGradient.data[0][col] = (col == currentDigit) ? -1 : 1;
            }
        }
    }

    private void instantiateParameters() {
        weights = new Matrix[layerCount];
        biases = new Matrix[layerCount];

        // initialize input layer weight/bias
        weights[0] = new Matrix(INPUT_NODES, neuronsPerHiddenLayer);
        biases[0] = new Matrix(1, neuronsPerHiddenLayer);

        // initialize intermediary matrices
        for(int i = 1; i < layerCount - 1; i++) {
            weights[i] = new Matrix(neuronsPerHiddenLayer, neuronsPerHiddenLayer);
            biases[i] = new Matrix(1, neuronsPerHiddenLayer);
        }

        // initialize last hidden layer weight/bias
        weights[layerCount - 1] = new Matrix(neuronsPerHiddenLayer, FINAL_LAYER_NEURONS);
        biases[layerCount - 1] = new Matrix(1, FINAL_LAYER_NEURONS);

        for(int i = 0; i < layerCount; i++) {
            weights[i].instantiateMatrix();
            biases[i].instantiateMatrix();
        }
    }

    public void feedForwardAndSetActivations(Matrix image) {
        activations = new Matrix[layerCount];

        // initialize first activation layer
        image = image.multiply(weights[0]);
        image.addAndSigmoid(biases[0], true);
        activations[0] = image;

        for(int i = 1; i < layerCount; i++) {
            // use previous activation layer to feed forward
            Matrix tempImage = activations[i-1].multiply(weights[i]);
            tempImage.addAndSigmoid(biases[i], true);
            activations[i] = tempImage;
        }
    }

    public Matrix feedForward(Matrix image) {
        for(int i = 0; i < layerCount; i++) {
            image = image.multiply(weights[i]);
            image.addAndSigmoid(biases[i], true);
        }
        return image;
    }

    public void train(int[] correct, Matrix[] trainingImages) {
        instantiateTrainingMatrices();
        for(int i = 0; i < trainingImages.length; i += BATCH_SIZE) {
            trainBatch(i, correct, trainingImages);
        }
    }


    private void trainBatch(int startIndex, int[] correct, Matrix[] trainingImages) {
        for(int i = startIndex; i < BATCH_SIZE; i++) {
            backprop(trainingImages[i], correct[i]);
        }
        updateWeightsAndBiases();
    }

    private void backprop(Matrix image, int correctDigit) {
        feedForwardAndSetActivations(image);
        recursiveBackprop(layerCount - 1, lastLayerActivationGradients[correctDigit]);
    }

    private void recursiveBackprop(int layerIndex, Matrix nextLayerGradient) {
        if(layerIndex < 0)
            return;

        Matrix currentActivationLayer = activations[layerIndex - 1];

        // calculate bias changes
        for(int col = 0; col < currentActivationLayer.data[0].length; col++) {
            double sigmoid = currentActivationLayer.data[0][col];
            double negativeGradient = (sigmoid * sigmoid) * ((1.0 / sigmoid) - 1) * -1;
            negativeGradient *= nextLayerGradient.data[0][col];
            biasChanges[layerIndex].data[0][col] += negativeGradient;
        }

        // calculate weight changes
        for(int col = 0; col < currentActivationLayer.data[0].length; col++) {
            double sigmoid = currentActivationLayer.data[0][col];
            double negativeGradient = (sigmoid * sigmoid) * ((1.0 / sigmoid) - 1) * -1;
            // todo: finish this
        }

        // calculate current layer gradient for next step of backprop
        if(layerIndex > 0) {
            Matrix currentActivationGradients = new Matrix(1, currentActivationLayer.data[0].length);
            // todo: finish this
            recursiveBackprop(layerIndex - 1, currentActivationGradients);
        }
    }

    private void updateWeightsAndBiases() {
        for(int i = 0; i < weights.length; i++) {
            weightChanges[i].scaleMatrix(1.0 / BATCH_SIZE);
            weights[i].addAndSigmoid(weightChanges[i], false);
            weightChanges[i].scaleMatrix(0); // reset weight change matrix
        }

        for(int i = 0; i < biases.length; i++) {
            biasChanges[i].scaleMatrix(1.0 / BATCH_SIZE);
            biases[i].addAndSigmoid(biasChanges[i], false);
            biasChanges[i].scaleMatrix(0); // reset bias change matrix
        }
    }

    private void instantiateTrainingMatrices() {
        for(int i = 0; i < weights.length; i++) {
            weightChanges[i] = new Matrix(weights[i].data.length, weights[i].data[0].length);
        }

        for(int i = 0; i < biases.length; i++) {
            biasChanges[i] = new Matrix(biases[i].data.length, biases[i].data[0].length);
        }
    }

    public double getTestAccuracy(int[] correct, Matrix[] testImages) {
        int networkScore = 0;
        for(int i = 0; i < testImages.length; i++) {
            testImages[i] = feedForward(testImages[i]);
            if(getMax(testImages[i]) == correct[i]) {
                networkScore++;
            }
        }

        return 1.0 * networkScore / correct.length;
    }

    private int getMax(Matrix outputMatrix) {
        if(outputMatrix.data.length != 1 || outputMatrix.data[0].length != 10) {
            throw new IllegalArgumentException("Feedforward output matrix has invalid dimensions!");
        }
        int maxDigit = 0;
        double currentMax = outputMatrix.data[0][0];

        for(int i = 1; i < 10; i++) {
            if(outputMatrix.data[0][i] > currentMax) {
                maxDigit = i;
                currentMax = outputMatrix.data[0][i];
            }
        }

        return maxDigit;
    }
}
