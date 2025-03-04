public class Matrix {
    double[][] data;

    public Matrix(int n, int m) {
        if(n <= 0 || m <= 0) {
            throw new IllegalArgumentException("A matrix must have positive dimensions.");
        }
        data = new double[n][m];
    }

    public void instantiateMatrix() {
        for(int i = 0; i < data.length; i++) {
            for(int j = 0; j < data[0].length; j++) {
                data[i][j] = 2 * Math.random() - 1;
            }
        }
    }

    public Matrix multiply(Matrix otherMatrix) {
        if(this.data[0].length != otherMatrix.data.length) {
            throw new IllegalArgumentException("Multiplication with invalid dimensions: A has " + this.data[0].length + " columns while B has " + otherMatrix.data.length + " rows.");
        }

        // A * B => new matrix has A's rows and B's columns
        Matrix newMatrix = new Matrix(this.data.length,  otherMatrix.data[0].length);

        for(int row = 0; row < this.data.length; row++) {
            for(int col = 0; col < otherMatrix.data[0].length; col++) {
                newMatrix.data[row][col] = dot(this, row, otherMatrix, col);
            }
        }

        return newMatrix;
    }

    private double dot(Matrix a, int row, Matrix b, int col) {
        double sum = 0;
        for(int i = 0; i < a.data[row].length; i++) {
            sum += a.data[row][i] * b.data[i][col]; // multiply row * column and sum
        }
        return sum;
    }

    public void addAndSigmoid(Matrix otherMatrix, boolean sigmoid) {
        if(this.data.length != otherMatrix.data.length || this.data[0].length != otherMatrix.data[0].length) {
            throw new IllegalArgumentException("Addition with invalid dimensions.");
        }

        for(int i = 0; i < data.length; i++) {
            for(int j = 0; j < data[0].length; j++) {
                data[i][j] += otherMatrix.data[i][j];
                if(sigmoid) {
                    data[i][j] = 1.0 / (1 + Math.exp(-data[i][j]));
                }
            }
        }
    }

    public void scaleMatrix(double factor) {
        for(int i = 0; i < data.length; i++) {
            for(int j = 0; j < data[0].length; j++) {
                data[i][j] *= factor;
            }
        }
    }
}
