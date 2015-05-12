package examples;

import java.text.DecimalFormat;
import java.util.Random;

/**
 * 用于显示LBFGS，采用一个线性回归进行测试
 * Created by yunshen on 2015/5/12.
 */
public class TestMain {
    private static double testFunction(double[] x) {
        double[] thea = {0.455, -0.36, 0.653};
        double result = 0;
        for (int i = 0; i < thea.length - 1; i++)
            result += Math.pow(x[i], 3) * thea[i];
        return result + thea[thea.length - 1];
    }

    public static void main(String[] args) {
        int sample_size = 1000;
        Random random = new Random();
        double[][] x = new double[sample_size][2];
        double[] y = new double[sample_size];
        for (int i = 0; i < sample_size; i++) {
            for (int j = 0; j < 2; j++) {
                x[i][j] = random.nextDouble();
            }
            y[i] = testFunction(x[i]);
        }
        int k = 3;
        double[] thea = new double[k];
        for (int i = 0; i < thea.length; i++) {
            thea[i] = 0;
        }
        DecimalFormat df = new DecimalFormat("#0.000");
        LinearRegression instance = new LinearRegression(x, y, thea);
        double[] result = instance.regression();
        for (int i = 0; i < result.length; i++) {
            System.out.print(df.format(result[i]) + "\t");
        }
        System.out.println();
    }
}
