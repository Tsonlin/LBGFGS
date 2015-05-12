package examples;


import main.LBFGS;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.matrix.Matrix;

import java.text.DecimalFormat;
import java.util.Date;

/**
 * 使用示例，利用LBFGS求解线性回归，重载损失函数、梯度函数以及回归入口
 */
public class LinearRegression extends LBFGS {
    private Logger logger = LoggerFactory.getLogger(LinearRegression.class);
    private double[][] x;
    private double[] y;

    public LinearRegression(double[][] x, double[] y, double[] thea) {
        this.x = x;
        this.y = y;
        this.theta = new Matrix(thea, thea.length);
    }

    /**
     * 重载损失函数
     * loss=1/2*∑[(f(xi)-yi)^2]
     */
    @Override
    public double costfunction() {
        double loss = 0;
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < theta.getRowDimension(); j++) {
                loss += Math.pow(evl(x[i]) - y[i], 2);
            }
        }
        loss = loss / 2;
        return loss;
    }

    /**
     * y=xi^3+b
     *
     * @param x
     * @return
     */
    private double evl(double[] x) {
        double result = 0;
        String xString = "";
//		display_matrix(theta.transpose());
        for (int i = 0; i < x.length; i++) {
            result += Math.pow(x[i], 3) * theta.get(i, 0);
            xString += "," + x[i];
        }
//		logger.debug("f("+xString.substring(1)+")="+result);
        return result + theta.get(theta.getRowDimension() - 1, 0);
    }

    /**
     * 重载梯度函数
     */
    @Override
    public Matrix gradient() {
//		display_matrix(theta,"theta");
        Matrix gradientMatrix = new Matrix(theta.getRowDimension(), 1, 0);
        for (int c = 0; c < y.length; c++) {
            Matrix tempMatrix = new Matrix(theta.getRowDimension(), 1, 0);
            for (int i = 0; i < theta.getRowDimension() - 1; i++)
                tempMatrix.set(i, 0, Math.pow(x[c][i], 3));
            tempMatrix.set(theta.getRowDimension() - 1, 0, 1);
            gradientMatrix.plusEquals(tempMatrix.times(evl(x[c]) - y[c]));
        }
//		display_matrix(gradientMatrix,"gradientMatrix");
        return gradientMatrix;
    }

    @Override
    public double[] regression() {
        logger.info("利用LBFGS开始训练分段评估函数！");
        Date start = new Date();
        batch_learningrate = 0.0001;
        Cost_threashold = 0.00001;
        linearLBFGS_descend_Batch(3);
        logger.info("LBFGS训练分段评估函数结束！耗时：" + new DecimalFormat("#0.00s").format((double) (new Date()).getTime() / start.getTime()));
        return theta.getColumnPackedCopy();
    }

}
