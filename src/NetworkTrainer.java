import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

public class NetworkTrainer {
    static final int PIXELS = 784;
    static final int TRAINING_IMAGES = 60000;
    static final int TEST_IMAGES = 10000;

    static final int HIDDEN_LAYERS = 2;
    static final int NEURONS_PER_HIDDEN_LAYER = 16;

    public static void main(String[] args) throws IOException {
        NeuralNetwork network = new NeuralNetwork(HIDDEN_LAYERS, NEURONS_PER_HIDDEN_LAYER);
        trainNetwork(network);
        testNetwork(network);
    }

    public static void trainNetwork(NeuralNetwork network) throws IOException {
        int[] trainingAnswers = new int[TRAINING_IMAGES];
        Matrix[] trainingImages = new Matrix[TRAINING_IMAGES];
        loadFileIntoMemory("mnist_train.csv", trainingAnswers, trainingImages);
        network.train(trainingAnswers, trainingImages);
    }

    public static void testNetwork(NeuralNetwork trainedNetwork) throws IOException {
        int[] testAnswers = new int[TEST_IMAGES];
        Matrix[] testImages = new Matrix[TEST_IMAGES];
        loadFileIntoMemory("mnist_test.csv", testAnswers, testImages);
        double accuracy = trainedNetwork.getTestAccuracy(testAnswers, testImages);
        System.out.printf("model achieved %.4f%% accuracy%n", accuracy*100);
    }

    public static void loadFileIntoMemory(String file, int[] correct, Matrix[] images) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        for(int i = 0; i < images.length; i++) {
            StringTokenizer st = new StringTokenizer(br.readLine(), ",");
            correct[i] = Integer.parseInt(st.nextToken()); // store the correct digit depicted by the image
            images[i] = new Matrix(1, PIXELS);
            for(int j = 0; j < PIXELS; j++) {
                images[i].data[0][j] = Integer.parseInt(st.nextToken()) / 255.0; // store the pixel intensity
            }
        }
    }
}
