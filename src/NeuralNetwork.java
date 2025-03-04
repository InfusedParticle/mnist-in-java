import java.io.IOException;

public class NeuralNetwork {
    int layerCount;
    int neuronsPerHiddenLayer;

    static final int INPUT_NODES = 28*28;
    static final int FINAL_LAYER_NEURONS = 10;

    Matrix[] weights;
    Matrix[] biases;

    static final int BATCH_SIZE = 10;
    static final int EPOCHS = 20;
    Matrix[] weightChanges;
    Matrix[] biasChanges;

    Matrix[] activations;

    public NeuralNetwork(int hiddenLayers, int neuronsPerHiddenLayer) {
        if(hiddenLayers <= 0 || neuronsPerHiddenLayer <= 0) {
            throw new IllegalArgumentException("Must have positive hidden layers and neurons per hidden layer");
        }
        this.layerCount = hiddenLayers + 1;
        this.neuronsPerHiddenLayer = neuronsPerHiddenLayer;
        instantiateParameters();
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
        activations = new Matrix[layerCount + 1]; // add 1 because image counts as activation

        // initialize first activation layer (image)
        activations[0] = image;

        for(int i = 0; i < layerCount; i++) {
            // use previous activation layer to feed forward
            Matrix tempImage = activations[i].multiply(weights[i]);
            tempImage.addAndSigmoid(biases[i], true);
            activations[i+1] = tempImage;
        }
    }

    public Matrix feedForward(Matrix image) {
        for(int i = 0; i < layerCount; i++) {
            image = image.multiply(weights[i]);
            image.addAndSigmoid(biases[i], true);
        }
        return image;
    }

    private void shuffleTrainingImagesAndCorrect(Matrix[] trainingImages, int[] correct) {
        for (int i = 0; i < trainingImages.length; i++) {
            int swapIndex = i + (int) (Math.random() * (trainingImages.length - i));
            swap(trainingImages, correct, i, swapIndex);
        }
    }

    private void swap(Matrix[] trainingImages, int[] correct, int i, int j) {
        Matrix temp = trainingImages[i];
        trainingImages[i] = trainingImages[j];
        trainingImages[j] = temp;

        int temp2 = correct[i];
        correct[i] = correct[j];
        correct[j] = temp2;
    }

    public void train(int[] correct, Matrix[] trainingImages) {
        instantiateTrainingMatrices();
        for(int iterations = 0; iterations < EPOCHS; iterations++) {
            System.out.println("Starting iteration " + iterations);
            shuffleTrainingImagesAndCorrect(trainingImages, correct);
            for(int i = 0; i < trainingImages.length; i += BATCH_SIZE) {
                trainBatch(i, correct, trainingImages);
            }
        }
    }

    private void trainBatch(int startIndex, int[] correct, Matrix[] trainingImages) {
        double batchCost = 0;
        for(int i = startIndex; i < startIndex + BATCH_SIZE; i++) {
            batchCost += backprop(trainingImages[i], correct[i]);
        }
        if(startIndex >= trainingImages.length - BATCH_SIZE - 1) {
            System.out.println("last batch cost: " + batchCost);
            try {
                NetworkTrainer.testNetwork(this);
            } catch(IOException e) {
                System.out.println("ioexception");
            }
        }
        updateWeightsAndBiases();
    }

    private double calculateCost(int correctDigit) {
        double cost = 0;
        for(int i = 0; i < 10; i++) {
            double predicted = activations[layerCount].data[0][i];
            if(i == correctDigit) {
                cost += (predicted - 1) * (predicted - 1);
            } else {
                cost += predicted * predicted;
            }
        }
        return cost;
    }

    private double backprop(Matrix image, int correctDigit) {
        feedForwardAndSetActivations(image);
        double cost = calculateCost(correctDigit);
        Matrix lastLayerGradient = calculateLastLayerGradient(correctDigit);
        recursiveBackprop(layerCount, lastLayerGradient);
        return cost;
    }

    private Matrix calculateLastLayerGradient(int correctDigit) {
        // calculates dC/dx where C is cost and x_i is the ith activation in the output layer
        // C = mean square error, sum((x_i - expected)^2)
        // expected == 1 if i is the correct digit, otherwise expected == 0
        // the partial derivative of C with respect to x_i is 2 * (x_i - expected)
        Matrix result = new Matrix(1, 10);
        for(int i = 0; i < 10; i++) {
            double costGradient;
            if(i == correctDigit) {
                costGradient = 2 * (activations[layerCount].data[0][i] - 1);
            } else {
                costGradient = 2 * activations[layerCount].data[0][i];
            }
            result.data[0][i] = costGradient;
        }
        return result;
    }

    private void recursiveBackprop(int layerIndex, Matrix currentLayerGradient) {
        Matrix currentActivationLayer = activations[layerIndex];
        Matrix previousActivationLayer = activations[layerIndex - 1];

        // calculate bias changes
        for(int col = 0; col < currentActivationLayer.data[0].length; col++) {
            double sigmoid = currentActivationLayer.data[0][col];
            // derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            double negativeGradient = sigmoid * (1 - sigmoid) * -1;
            negativeGradient *= currentLayerGradient.data[0][col];
            biasChanges[layerIndex - 1].data[0][col] += negativeGradient;
        }

        // calculate weight changes
        for(int col = 0; col < currentActivationLayer.data[0].length; col++) {
            double sigmoid = currentActivationLayer.data[0][col];
            double negativeGradient = sigmoid * (1 - sigmoid) * -1;
            negativeGradient *= currentLayerGradient.data[0][col];
            int activationsInPrevLayer = previousActivationLayer.data[0].length;
            for(int weightIndex = 0; weightIndex < activationsInPrevLayer; weightIndex++) {
                // scale weight gradient by its corresponding activation
                double gradientFactor = previousActivationLayer.data[0][weightIndex];
                weightChanges[layerIndex - 1].data[weightIndex][col] += gradientFactor * negativeGradient;
            }
        }

        // calculate previous layer gradient for next step of backprop
        // currentLayerGradient * currentActivationGradient = previousLayerGradient

        // currentActivationGradient is a matrix of dx/da for all a_j and x_i
        // a_j is a previous layer activation and x_i is a current layer activation
        if(layerIndex > 1) {
            int currentActivations = currentActivationLayer.data[0].length;
            int previousActivations = previousActivationLayer.data[0].length;
            Matrix currentActivationGradients = new Matrix(currentActivations, previousActivations);
            for(int i = 0; i < currentActivations; i++) {
                double sigmoid = currentActivationLayer.data[0][i];
                double gradient = (sigmoid) * (1 - sigmoid);
                for(int j = 0; j < previousActivations; j++) {
                    double gradientFactor = weights[layerIndex - 1].data[j][i];
                    currentActivationGradients.data[i][j] = gradient * gradientFactor;
                }
            }
            Matrix previousLayerGradient = currentLayerGradient.multiply(currentActivationGradients);
            recursiveBackprop(layerIndex - 1, previousLayerGradient);
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
        weightChanges = new Matrix[weights.length];
        for(int i = 0; i < weights.length; i++) {
            weightChanges[i] = new Matrix(weights[i].data.length, weights[i].data[0].length);
        }

        biasChanges = new Matrix[biases.length];
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
