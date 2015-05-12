package main;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.matrix.Matrix;

/**
 * Created by Bys on 2015/3/25.
 */
public abstract class LBFGS {
    private static Logger logger = LoggerFactory.getLogger(LBFGS.class);

    protected Matrix theta = null;
    // Matrix x = null;
    // Matrix y = null;
//	protected double batch_learningrate = 0.00003;
    protected double batch_learningrate = 0.00001;
    protected double Max_Iter_Num = 20000;
    protected double Cost_threashold_Rate = 0.0000001;
    protected double Cost_threashold = 1;
    protected Matrix L_BFGS_Hessian = null;
    protected Matrix L_BFGS_y = null;
    protected Matrix L_BFGS_s = null;
    protected int L_BFGS_k = 0;
    protected double[] lou_vector = null;


    protected void display_matrix(Matrix matrix, String name) {
        logger.debug("----------" + name + "----------start--------");
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            String line = "";
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                line += "\t" + matrix.get(i, j);
            }
            logger.debug(line);
        }
        logger.debug("----------" + name + "----------end--------");
    }

    // y=theta' * x

    /**
     * 返回损失函数
     */
    abstract public double costfunction();

    private int searchOneDimension(Matrix dk) {
        int count = 1;
        double least = costfunction();
        Matrix x_matrix = theta;
        int leastIndex = 1;
        while (count < 1000) {
            Matrix dkTmp = dk.times(batch_learningrate * count);
            theta.minusEquals(dkTmp);
            double tmpCost = costfunction();
            if (tmpCost < least) {
                x_matrix = theta;
                leastIndex = count;
            }
            count++;
        }
        theta = x_matrix;
        return count;
    }

    /**
     * 梯度向量 对每个样本执行Tao(J)/Tao(theta(j)) ，对线性回归 =(theta * x-y)* xj
     */
    abstract public Matrix gradient();

    public void linearLBFGS_descend_Batch(int m) {
        L_BFGS_y = new Matrix(theta.getRowDimension(), m, 0);
        L_BFGS_s = new Matrix(theta.getRowDimension(), m, 0);
        lou_vector = new double[m];
        for (int i = 0; i < m; i++)
            lou_vector[i] = 0;
        L_BFGS_Hessian = Matrix.identity(theta.getRowDimension(),
                theta.getRowDimension());

        double errcost_last = costfunction();
        double cost_Rate = 1000;
        logger.info("ErrCost(0):" + errcost_last);
        int errCount = 0;
        for (int it = 0; it < Max_Iter_Num; it++) {
            Matrix g1 = gradient();
//			display_matrix(g1);
//			display_matrix(L_BFGS_Hessian);
            Matrix dk = L_BFGS_Hessian.times(g1).times(batch_learningrate);// dk
            // is
            // xk+1-
            // xk
//			dk = dk.times(searchOneDimension(dk));
            theta.minusEquals(dk);

            Matrix g2 = gradient();
            Matrix delta_gradient = g2.minus(g1);
            L_BFGS_k++;
//			 display_matrix(delta_gradient,"delta_gradient");
//			 display_matrix(dk,"dk");
            Hessian(delta_gradient, dk, m);
//			 display_matrix(L_BFGS_Hessian);
//			display_matrix(theta.transpose());
            double errcost = costfunction();
            cost_Rate = (errcost_last - errcost) / errcost_last;
            errcost_last = errcost;
            logger.info("ErrCost(" + (it + 1) + "):" + errcost);
            double[] result = theta.getColumnPackedCopy();
//			for(int i=0;i<result.length;i++){
//				System.out.print((new DecimalFormat("#0.000000")).format(result[i])+"\t");
//			}
//			System.out.println();
//			display_matrix(theta.transpose(), "theat");
            if (cost_Rate < 0) {
                errCount++;
            } else {
                errCount = 0;
            }
            if (errCount > 5)
                return;
            if (errcost < Cost_threashold)
                return;
        }
    }

    /**
     * v'*M*v= V为N*m为矩阵
     */
    public Matrix vtMvMatrix(Matrix v, Matrix M) {
        Matrix Middle = v.transpose().times(M).times(v);
        return Middle;
    }

    /**
     * L_BFGS 的头项
     *
     * @param y N*m维的梯度差分矩阵
     * @param s N*m维的的待求解向量差分矩阵
     * @param n 记忆次数
     */
    public Matrix Hessian_Header(Matrix y, Matrix s, int n) {
        int[] index_set = {0};
        int m = 0;
        if (L_BFGS_k <= n - 1) {
            m = L_BFGS_k;
        } else {
            m = n;
        }
        Matrix I = Matrix.identity(y.getRowDimension(), y.getRowDimension());
        Matrix vector_y = y.getMatrix(0, y.getRowDimension() - 1, index_set);
        Matrix vector_s = s.getMatrix(0, s.getRowDimension() - 1, index_set);
        Matrix v = I.minus(vector_y.times(vector_s.transpose()).times(
                lou_vector[0]));
        Matrix Header = vtMvMatrix(v, I);// v0'Hv0
        for (int i = 1; i < m; i++) {
            index_set[0] = i;
            vector_y = y.getMatrix(0, y.getRowDimension() - 1, index_set);
            vector_s = s.getMatrix(0, s.getRowDimension() - 1, index_set);
            v = I.minus(vector_y.times(vector_s.transpose()).times(
                    lou_vector[i]));
            Header = vtMvMatrix(v, Header);
        }
//		Header.times(vector_s.transpose().times(vector_y).get(0, 0)/vector_y.transpose().times(vector_y).get(0, 0));
        return Header;
    }

    /**
     * L_BFGS 的尾项
     *
     * @param y N*m维的梯度差分矩阵
     * @param s N*m维的的待求解向量差分矩阵
     * @param n 记忆次数
     */
    public Matrix Hessian_Tail(Matrix y, Matrix s, int n) {
        int m = 0;
        if (L_BFGS_k <= n - 1) {
            m = L_BFGS_k;
        } else {
            m = n;
        }
        int[] index_set = {m - 1};
        Matrix I = Matrix.identity(y.getRowDimension(), y.getRowDimension());
        Matrix vector_s = s.getMatrix(0, s.getRowDimension() - 1, index_set);
        Matrix Tail = vector_s.times(vector_s.transpose()).times(
                lou_vector[index_set[0]]);
        for (int j = 0; j < m - 2; j++)// for each item ,s0,.....sk-1 as middle
        {
            index_set[0] = j;
            vector_s = s.getMatrix(0, s.getRowDimension() - 1, index_set);
            Matrix Middle = vector_s.times(vector_s.transpose()).times(
                    lou_vector[j]);
            for (int i = j + 1; i < m - 2; i++)// for each v,v1....vk
            {
                index_set[0] = i;
                Matrix vector_v_s = s.getMatrix(0, s.getRowDimension() - 1,
                        index_set);
                Matrix vector_v_y = y.getMatrix(0, y.getRowDimension() - 1,
                        index_set);
                Matrix v = I.minus(vector_v_s.times(vector_v_y.transpose())
                        .times(lou_vector[0]));
                Middle = vtMvMatrix(v, Middle);// ∏ v'*middle*v
            }
            Tail.plus(Middle);// ∑ each item
            // v(k)'*....*v(1)'*middle*v(1)....*v(k)
        }
        return Tail;
    }

    /**
     * L_BFGS 主要迭代开始
     *
     * @param vector_y 第k次的梯度差分
     * @param vector_s 第k次的待求解向量差分
     * @param n        记忆次数
     */
    public void Hessian(Matrix vector_y, Matrix vector_s, int n) {
        refresh_s_y(vector_y, vector_s);
        refresh_lou_vector(vector_y, vector_s);
        L_BFGS_Hessian = Hessian_Header(L_BFGS_y, L_BFGS_s, n).plus(
                Hessian_Tail(L_BFGS_y, L_BFGS_s, n));
    }

    private void refresh_s_y(Matrix vector_y, Matrix vector_s) {
        int m = L_BFGS_y.getColumnDimension();
        int index = L_BFGS_k;
        if (L_BFGS_k > m) {
            index = m;
            L_BFGS_y.setMatrix(0, vector_y.getRowDimension() - 1, 0, m - 2,
                    L_BFGS_y.getMatrix(0, vector_y.getRowDimension() - 1, 1, m - 1));
            L_BFGS_s.setMatrix(0, vector_s.getRowDimension() - 1, 0, m - 2,
                    L_BFGS_s.getMatrix(0, vector_s.getRowDimension() - 1, 1, m - 1));
        }
        L_BFGS_y.setMatrix(0, vector_y.getRowDimension() - 1, index - 1, index - 1,
                vector_y);
        L_BFGS_s.setMatrix(0, vector_s.getRowDimension() - 1, index - 1, index - 1,
                vector_s);
//		display_matrix(L_BFGS_y,"L_BFGS_y");
//		display_matrix(L_BFGS_s,"L_BFGS_s");
    }

    private void refresh_lou_vector(Matrix vector_y, Matrix vector_s) {
        int m = L_BFGS_y.getColumnDimension();
        int index = L_BFGS_k;
        if (L_BFGS_k >= m - 1) {
            index = m - 1;
            for (int i = 0; i < m - 2; i++) {
                lou_vector[i] = lou_vector[i + 1];
            }
        }
        lou_vector[index] = (double) 1
                / vector_y.transpose().times(vector_s).get(0, 0);
    }


    public double getBatch_learningrate() {
        return batch_learningrate;
    }

    public void setBatch_learningrate(double batch_learningrate) {
        this.batch_learningrate = batch_learningrate;
    }

    public double getMax_Iter_Num() {
        return Max_Iter_Num;
    }

    public void setMax_Iter_Num(double max_Iter_Num) {
        Max_Iter_Num = max_Iter_Num;
    }

    public double getCost_threashold_Rate() {
        return Cost_threashold_Rate;
    }

    public void setCost_threashold_Rate(double cost_threashold_Rate) {
        Cost_threashold_Rate = cost_threashold_Rate;
    }

    public double getCost_threashold() {
        return Cost_threashold;
    }

    public void setCost_threashold(double cost_threashold) {
        Cost_threashold = cost_threashold;
    }

    abstract public double[] regression();

}
