package lab2020;
import Jama.Matrix;

import java.io.*;
import java.util.*;
import java.lang.*;
public class RankObsModel<T>  implements Serializable{
    int seed=1;
    Random random = new Random(seed);

    public T log_file;
    public ArrayList<List<Double>> train_data;
    public int num_users;
    public int num_skills;
    public int num_questions;
    public int num_examples;
    public int num_attempts;
    public int num_concepts;
    public T lambda_s;
    public T lambda_t;
    public T lambda_q;
    public T lambda_bias;
    public T penalty_weight;
    public T markovian_steps;
    public double lr;
    public T tol;
    public T max_iter;

    public boolean binarized_question = false;
    public boolean exact_penalty = false;
    public boolean log_sigmoid = false;
    public T current_test_attempt = null;
    public List<Double> loss_list;

    public List<List<Double>>train_order;
    public ArrayList<Double> all_obs ;

    public double[][] S ;
    public double[][][] T ;
    public double[][] Q ;

    public double[][][]obs_tensor;

    public double[]bias_s;
    public double[]bias_q;
    public double[]bias_a;

    public double global_avg;


    public RankObsModel(Config config){
        System.out.println(String.format("seed %d",seed));
        this.log_file=(T)config.log_file;
        this.train_data=(ArrayList<List<Double>>)config.train;
        this.num_users=((Long)(Object)config.num_users).intValue();
        this.num_attempts=((Long)(Object)config.num_attempts).intValue();
        this.num_questions=((Long)(Object)config.num_questions).intValue();
        this.num_skills=((Double)(Object)config.num_skills).intValue();
        this.num_concepts=((Double)(Object)config.num_concepts).intValue();
        this.num_examples=((Long)(Object)config.num_discussions).intValue();
        this.lambda_s=(T)config.lambda_s;
        this.lambda_t=(T)config.lambda_t;
        this.lambda_q=(T)config.lambda_q;
        this.lambda_bias=(T)config.lambda_bias;
        this.penalty_weight=(T)config.penalty_weight;
        this.markovian_steps=(T)config.markovian_steps;
        this.lr=(double)config.lr;
        this.tol=(T)config.tol;
        this.max_iter=(T)config.max_iter;

//        System.out.println(this.num_skills.getClass());
//        this.num_skills=(int)(Object)this.num_skills;
//        System.out.println(this.num_users.getClass());

        train_order_and_all_obs();

        this.S=new double[this.num_users][this.num_skills];
        two_d_matrix(this.S,this.num_users,this.num_skills);
        this.T=new double[this.num_skills][this.num_attempts][this.num_concepts];
        three_d_matrix(this.T,this.num_skills,this.num_attempts,this.num_concepts);
        this.Q=new double[this.num_concepts][this.num_questions];
        two_d_matrix(this.Q,this.num_concepts,this.num_questions);

        this.obs_tensor=new double[this.num_users][this.num_attempts][this.num_questions];
        //three_d_matrix(this.obs_tensor,this.num_users,this.num_attempts,this.num_questions);
        initializing_tensor_obs();

        this.bias_s=new double[this.num_users];
        this.bias_q=new double[this.num_questions] ;
        this.bias_a=new double[this.num_attempts];

        double temp=0;
        for(int i=0;i<this.all_obs.size();i++){
            temp+=all_obs.get(i);
        }
        this.global_avg=temp/this.all_obs.size();
        System.out.println(String.format("global_avg: %f",this.global_avg));

    }
    //for creating S T Q
    public void one_d_matrix(double[]data,int s1){
        //data=new double[s1];
        for(int i=0;i<s1;i++){
            data[i]=random.nextDouble();;
        }
    }
    public void two_d_matrix(double[][]data,int s1,int s2){
        //data=new double[s1][s2];
        for(int i=0;i<s1;i++)
        {
            for(int j=0;j<s2;j++)
            {
                data[i][j]=random.nextDouble();;
            }
        }
    }
    public void three_d_matrix(double[][][]data,int s1,int s2,int s3){
        // data=new double[s1][s2][s3];
        for(int i=0;i<s1;i++)
        {
            for(int j=0;j<s2;j++)
            {
                for(int k=0;k<s3;k++)
                {
                    data[i][j][k]= random.nextDouble();
                }
            }
        }
    }

     //    size*3, size*1;stu,att,que,obs
     //   rank_obs.py line68-75 for loop
     public void train_order_and_all_obs()
     {
          this.all_obs=new ArrayList<Double>();
          this.train_order=new ArrayList<List<Double>>();
          for(int i=0; i< this.train_data.size() ; i++)
          {
             // System.out.println(this.train_data.get(i).get(3));
                this.all_obs.add(this.train_data.get(i).get(3));//obs
                this.train_order.add(new ArrayList<>(this.train_data.get(i).subList(0,3)));// stu, att, que


//              this.train_order[i][0]=(double)(Object)this.train_data[i][0];//stu
//              this.train_order[i][1]=(double)(Object)this.train_data[i][1];//att
//              this.train_order[i][2]=(double)(Object)this.train_data[i][2];//que
              double range=Math.max(0,this.train_data.get(i).get(1)-(double)(Object)this.markovian_steps);
             // System.out.println("max"+range+this.train_data.get(i).get(1));
              for(int j=(int)range;j<this.train_data.get(i).get(1);j++)
              {
                    List<Double> tmp=new ArrayList<Double>();
                    tmp.add(this.train_data.get(i).get(0));
                    tmp.add((double)j);
                    tmp.add(this.train_data.get(i).get(2));
                    this.train_order.add(tmp);
              }
              range=Math.min(this.num_attempts,
                      this.train_data.get(i).get(1)+(double)(Object)this.markovian_steps+1);
              for(int j=(int)(double)this.train_data.get(i).get(1)+1;j<range;j++) {
                  List<Double> tmp = new ArrayList<Double>();
                  tmp.add(this.train_data.get(i).get(0));
                  tmp.add((double) j);
                  tmp.add(this.train_data.get(i).get(2));
                  this.train_order.add(tmp);
              }
          }
     }

     //rank_obs.py line96-100
     public void initializing_tensor_obs()
     {
         for(int i=0; i< this.train_data.size() ; i++)
         {
             double obs=this.train_data.get(i).get(3);
             if(obs==0.0){
                 obs=1e-6;
             }
             //System.out.println(this.obs_tensor[1][1][2]);
             this.obs_tensor[(int)(double)this.train_data.get(i).get(0)]
                     [(int)(double)this.train_data.get(i).get(1)]
                     [(int)(double)this.train_data.get(i).get(2)]
                     =obs;
         }
     }

     public static double sigmoid(double x,int derivative_flag){
        double sigm;
        if(x>100){
                sigm=1;
        }
        else if(x<-100){
            sigm=0;
        }
        else {
            sigm = 1. / (1. + Math.exp(-x));
        }

         if (derivative_flag==1) {
             return sigm * (1. - sigm);
         }
         return sigm;
     }
    public static double[][] dot_product(double [][]a,double [][]b){
        double [][]result=new double[a.length][b[0].length];
        for (int i=0; i<a.length;i++)
        {
            for (int j=0;j<b[0].length;j++)
            {
                result[i][j]=0;
                for(int k=0;k<b.length;k++)
                {
                    result[i][j]+=a[i][k]*b[k][j];
                }
            }
        }
        return result;
    }
    public static double[][] outer_product(double a[][],double b[][]){
        double [][]result=new double[a[0].length][b.length];
        for (int i=0; i<a[0].length;i++)
        {
            for (int j=0;j<b.length;j++)
            {

                result[i][j]=a[0][i]*b[j][0];

            }
        }
        return result;
    }

    public static double[][] array_slice_2d(double [][]data, int left_right,int row_column){
        double [][]result;
        if(left_right==0){
            result=new double[1][data[0].length];
            for(int i=0;i<data[0].length;i++){
                result[0][i]=data[row_column][i];
            }
            return result;
        }
        else {
            result=new double[data.length][1];
            for(int i=0;i<data.length;i++){
                result[i][0]=data[i][row_column];
            }
            return result;
        }


    }

    public static double[][] array_slice_3d(double [][][]data,int row_column){
        double [][]result=new double[data.length][data[0][0].length];
        for(int i=0;i<data.length;i++){
            for (int j=0;j<data[0][0].length;j++){
                result[i][j]=data[i][row_column][j];
            }
        }
        return result;
    }

    public static double[][] array_to_matrix(double []data){
        double[][] result=new double[1][data.length];
        for (int i=0;i<data.length;i++){
            result[0][i]=data[i];
        }
        return result;
    }

    public  double _get_prediction(List<Double> one){
            double[][] pred_matrix = dot_product(
                    dot_product(array_slice_2d(this.S, 0, one.get(0).intValue()),
                    array_slice_3d(this.T, one.get(1).intValue())
                    ),array_slice_2d(this.Q, 1, one.get(2).intValue()));
            double pred=pred_matrix[0][0];
//            System.out.println(one.get(0).intValue());
//        System.out.println(this.bias_s.length);
            pred+=this.bias_s[one.get(0).intValue()];
            pred+=this.bias_q[one.get(2).intValue()];
            pred+=this.bias_a[one.get(1).intValue()];
            pred+=this.global_avg;

        if (this.binarized_question){
            pred = sigmoid(pred,0);
        }
            return pred;
     }

     public  double _get_penalty(){
        double penalty = 0.0;
        if((double)(Object)this.penalty_weight!=0.0){
            for(int i=0;i<this.train_order.size();i++){
                List<Double> one=new ArrayList<>(this.train_order.get(i).subList(0,3));
                if(one.get(1)>=1){
                    Matrix one_matrix=new Matrix(array_slice_3d(this.T,one.get(1).intValue()));
                    Matrix two_matrix=new Matrix(array_slice_3d(this.T,(one.get(1).intValue()-1)));
                    Matrix gap=one_matrix.minus(two_matrix);
                    double[][] tmp=gap.getArrayCopy();


                    for(int x=0;x<tmp.length;x++){
                        for (int y=0;y<tmp[0].length;y++){
                            if(tmp[x][y]>0) {
                                tmp[x][y]=0;
                            }
                        }
                    }
                    double diff=0;
                    if(this.exact_penalty)
                    {
                        for(int x=0;x<tmp.length;x++)
                        {
                            for (int y=0;y<tmp[0].length;y++)
                            {
                                diff+=tmp[x][y];
                            }
                        }
                    }
                    else{

                        diff=dot_product(
                                dot_product(array_slice_2d(this.S,0,one.get(0).intValue()),tmp)
                                ,array_slice_2d(this.Q,1,one.get(2).intValue()))[0][0];
                    }
                    if(this.log_sigmoid){
                        diff = Math.log(sigmoid(diff,0));
                    }
                    penalty-=(double)(Object)this.penalty_weight*diff;

                }
            }

        }
        return penalty;
     }

     public  double[] _get_loss(){
         double loss=0.0;double square_loss = 0.0;double bias_reg = 0.0;
         for(int i=0;i<this.train_data.size();i++)
         {
             double pred=_get_prediction(this.train_data.get(i).subList(0,3));
             square_loss+=Math.pow((this.train_data.get(i).get(3)-pred),2);
         }
         Matrix S = new Matrix(this.S);
         double reg_S=Math.pow(S.normF(),2);
         Matrix Q = new Matrix(this.Q);
         double reg_Q=Math.pow(Q.normF(),2);

         double[][]tmp=new double[1][this.T.length];
         for(int i=0;i<this.T.length;i++){
             Matrix T_tmp = new Matrix(this.T[i]);
             double reg_tmp=T_tmp.normF();
             tmp[0][i]=reg_tmp;
         }
         Matrix T = new Matrix(tmp);
         double reg_T=Math.pow(T.normF(),2);

         double reg_loss=(double)(Object)this.lambda_s*reg_S+(double)(Object)this.lambda_q*reg_Q+(double)(Object)this.lambda_t*reg_T;

         loss=square_loss+reg_loss;

         double rmse=Math.sqrt(square_loss/this.train_data.size());

         if ((double)(Object)this.lambda_bias!=0.0){
                Matrix bias__s=new Matrix(array_to_matrix(this.bias_s));
                Matrix bias__q=new Matrix(array_to_matrix(this.bias_q));
                Matrix bias__a=new Matrix(array_to_matrix(this.bias_a));
                bias_reg=(double)(Object)this.lambda_bias*(
                        Math.pow(bias__s.normF(),2)+ Math.pow(bias__q.normF(),2)+ Math.pow(bias__a.normF(),2)
                        );
         }

         double penalty=_get_penalty();

         loss+=bias_reg+penalty;
         double result[]={loss,rmse,reg_loss,penalty,bias_reg};
            return result;
     }

     public double[][] _grad_S_k(List<Double> one,double obs,int obs_flag){
            double grad[][]=new double[1][this.S[one.get(0).intValue()].length];
            //two_d_matrix(grad,1,this.S[one.get(0).intValue()].length);
         Matrix grad_tmp=new Matrix(grad);

            if(obs_flag!=0)
            {
                double pred=_get_prediction(one);
                if(this.binarized_question){
                    Matrix tmp=new Matrix(dot_product(array_slice_3d(this.T, one.get(1).intValue()),array_slice_2d(this.Q, 1, one.get(2).intValue())));
                    grad_tmp=tmp.transpose().times((1-pred)*pred*(obs-pred)*(-2));
                    //grad=tmp.getArrayCopy();
                }
                else{
                    Matrix tmp=new Matrix(dot_product(array_slice_3d(this.T, one.get(1).intValue()),array_slice_2d(this.Q, 1, one.get(2).intValue())));
                    grad_tmp=tmp.transpose().times((obs-pred)*(-2));
                    //grad=tmp.getArrayCopy();
                    //System.out.println(grad_tmp.getRowDimension());
                }
            }

            Matrix tmp=new Matrix(array_slice_2d(this.S, 0, one.get(0).intValue()));
            tmp=tmp.times(2*(double)(Object)this.lambda_s);
//            System.out.println(tmp.getRowDimension());
//         System.out.println(grad_tmp.getRowDimension());
            grad_tmp=grad_tmp.plus(tmp);

            if((double)(Object)this.penalty_weight!=0.0)
            {
                if(!this.exact_penalty)
                {
                    Matrix diff;
                    if(one.get(1)==0.0)
                    {
                        Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()+1));
                        Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                        diff=T_tmp_1.minus(T_tmp_2);
                    }
                    else if(one.get(1)==((double)this.num_attempts-1.0))
                    {
                        Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                        Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()-1));
                        diff=T_tmp_1.minus(T_tmp_2);
                    }
                    else
                        {
                            Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                            Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()-1));
                            diff=T_tmp_1.minus(T_tmp_2);
                            T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()+1));
                            diff=diff.plus(T_tmp_2.minus(T_tmp_1));
                        }
                    double [][]diff_array=diff.getArrayCopy();
                    for(int i=0;i<diff_array.length;i++){
                        for(int j=0;j<diff_array[0].length;j++){
                            if(diff_array[i][j]>0){
                                diff_array[i][j]=0;
                            }
                        }
                    }
                    Matrix val=new Matrix(dot_product(diff_array,array_slice_2d(this.Q,1,one.get(2).intValue())));
                    grad_tmp=grad_tmp.minus(val.transpose().times((-1)*(double)(Object)this.penalty_weight));
                }
            }
            grad=grad_tmp.getArrayCopy();
            return grad;
    }
    public double[][] _grad_T_ij(List<Double> one,double obs,int obs_flag){
        double grad[][]=new double[this.T.length] [this.T[0][one.get(2).intValue()].length];
        Matrix grad_tmp=new Matrix(grad);
        if(obs_flag!=0)
        {
            double pred=_get_prediction(one);
            if(this.binarized_question)
            {
                Matrix tmp=new Matrix(outer_product(array_slice_2d(this.S,0,one.get(0).intValue()),array_slice_2d(this.Q,1,one.get(2).intValue())));
                grad_tmp=tmp.times((1-pred)*pred*(obs-pred)*(-2));
                //grad=tmp.getArrayCopy();
            }
            else
                {
                Matrix tmp=new Matrix(outer_product(array_slice_2d(this.S,0,one.get(0).intValue()),array_slice_2d(this.Q,1,one.get(2).intValue())));
                grad_tmp=tmp.times((obs-pred)*(-2));
                //grad=tmp.getArrayCopy();
            }
        }

        Matrix tmp=new Matrix(array_slice_3d(this.T,one.get(1).intValue()));
        tmp=tmp.times(2*(double)(Object)this.lambda_t);
        grad_tmp=grad_tmp.plus(tmp);

        if((double)(Object)this.penalty_weight!=0.0)
        {
            Matrix diff;
            if(one.get(1)==0.0)
            {
                Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()+1));
                Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                diff=T_tmp_1.minus(T_tmp_2);

                double [][]diff_array=diff.getArrayCopy();
                for(int i=0;i<diff_array.length;i++){
                    for(int j=0;j<diff_array[0].length;j++){
                        if(diff_array[i][j]>0){
                            diff_array[i][j]=0;
                        }
                    }
                }

                if(this.exact_penalty)
                {
                    for(int i=0;i<diff_array.length;i++){
                        for(int j=0;j<diff_array[0].length;j++){
                            if(diff_array[i][j]<0){
                                diff_array[i][j]=-1;
                            }
                        }
                    }
                    diff=new Matrix(diff_array);
                    grad_tmp=grad_tmp.minus(diff.times((double)(Object)this.penalty_weight));
                }
                else{
                    diff_array=dot_product(
                            dot_product(array_slice_2d(this.S,0,one.get(0).intValue()), diff_array),
                            array_slice_2d(this.Q,1,one.get(2).intValue()));
                    tmp=new Matrix(outer_product(
                            array_slice_2d(this.S,0,one.get(0).intValue()),
                            array_slice_2d(this.Q,1,one.get(2).intValue())));
                    grad_tmp=grad_tmp.minus(tmp.times((double)(Object)this.penalty_weight*diff_array[0][0]));
                }
            }
            else if(one.get(1)==((double)this.num_attempts-1))
            {
                Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()-1));
                diff=T_tmp_1.minus(T_tmp_2);

                double [][]diff_array=diff.getArrayCopy();
                for(int i=0;i<diff_array.length;i++){
                    for(int j=0;j<diff_array[0].length;j++){
                        if(diff_array[i][j]>0){
                            diff_array[i][j]=0;
                        }
                    }
                }

                if(this.exact_penalty)
                {
                    for(int i=0;i<diff_array.length;i++){
                        for(int j=0;j<diff_array[0].length;j++){
                            if(diff_array[i][j]<0){
                                diff_array[i][j]=1;
                            }
                        }
                    }
                    diff=new Matrix(diff_array);
                    grad_tmp=grad_tmp.minus(diff.times((double)(Object)this.penalty_weight));
                }
                else
                    {
                        diff_array=dot_product(
                                dot_product(array_slice_2d(this.S,0,one.get(0).intValue()), diff_array),
                                array_slice_2d(this.Q,1,one.get(2).intValue()));
                        tmp=new Matrix(outer_product(
                                array_slice_2d(this.S,0,one.get(0).intValue()),
                                array_slice_2d(this.Q,1,one.get(2).intValue())));
                        grad_tmp=grad_tmp.minus(tmp.times((-1)*(double)(Object)this.penalty_weight*diff_array[0][0]));
                    }
            }
            else
                {
                    Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                    Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()-1));
                    diff=T_tmp_1.minus(T_tmp_2);

                    double [][]diff_array=diff.getArrayCopy();
                    for(int i=0;i<diff_array.length;i++){
                        for(int j=0;j<diff_array[0].length;j++){
                            if(diff_array[i][j]>0){
                                diff_array[i][j]=0;
                            }
                        }
                    }
                    if(this.exact_penalty)
                    {
                        for(int i=0;i<diff_array.length;i++){
                            for(int j=0;j<diff_array[0].length;j++){
                                if(diff_array[i][j]<0){
                                    diff_array[i][j]=1;
                                }
                            }
                        }
                        diff=new Matrix(diff_array);
                        grad_tmp=grad_tmp.minus(diff.times((double)(Object)this.penalty_weight));
                    }
                    else
                    {
                        diff_array=dot_product(
                                dot_product(array_slice_2d(this.S,0,one.get(0).intValue()), diff_array),
                                array_slice_2d(this.Q,1,one.get(2).intValue()));
                        tmp=new Matrix(outer_product(
                                array_slice_2d(this.S,0,one.get(0).intValue()),
                                array_slice_2d(this.Q,1,one.get(2).intValue())));
                        grad_tmp=grad_tmp.minus(tmp.times((-1)*(double)(Object)this.penalty_weight*diff_array[0][0]));
                    }

                     T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()+1));
                    diff=T_tmp_2.minus(T_tmp_1);
                    diff_array=diff.getArray();
                    for(int i=0;i<diff_array.length;i++){
                        for(int j=0;j<diff_array[0].length;j++){
                            if(diff_array[i][j]>0){
                                diff_array[i][j]=0;
                            }
                        }
                    }

                    if(this.exact_penalty)
                    {
                        for(int i=0;i<diff_array.length;i++){
                            for(int j=0;j<diff_array[0].length;j++){
                                if(diff_array[i][j]<0){
                                    diff_array[i][j]=-1;
                                }
                            }
                        }
                        diff=new Matrix(diff_array);
                        grad_tmp=grad_tmp.minus(diff.times((double)(Object)this.penalty_weight));
                    }
                    else
                    {
                        diff_array=dot_product(
                                dot_product(array_slice_2d(this.S,0,one.get(0).intValue()), diff_array),
                                array_slice_2d(this.Q,1,one.get(2).intValue()));
                        tmp=new Matrix(outer_product(
                                array_slice_2d(this.S,0,one.get(0).intValue()),
                                array_slice_2d(this.Q,1,one.get(2).intValue())));
                        grad_tmp=grad_tmp.minus(tmp.times((double)(Object)this.penalty_weight*diff_array[0][0]));
                    }


                }
        }
        grad=grad_tmp.getArrayCopy();
        return grad;
    }
    public double[][] _grad_Q_k(List<Double> one,double obs,int obs_flag){
        double grad[][]=new double[this.Q.length][1];
        //two_d_matrix(grad,1,this.S[one.get(0).intValue()].length);
        Matrix grad_tmp=new Matrix(grad);
        if(obs_flag!=0)
        {
            double pred=_get_prediction(one);
            if(this.binarized_question){
                Matrix tmp=new Matrix(dot_product(
                        array_slice_2d(this.S, 0, one.get(0).intValue()),
                        array_slice_3d(this.T, one.get(1).intValue())
                )
                );
                grad_tmp=tmp.transpose().times((1-pred)*pred*(obs-pred)*(-2));
                //grad=tmp.getArrayCopy();
            }
            else{
                Matrix tmp=new Matrix(dot_product(
                        array_slice_2d(this.S, 0, one.get(0).intValue()),
                        array_slice_3d(this.T, one.get(1).intValue())
                )
                );
                grad_tmp=tmp.transpose().times((1-pred)*(obs-pred)*(-2));
            }
        }

        Matrix tmp1=new Matrix(array_slice_2d(this.Q, 1, one.get(2).intValue()));
        tmp1=tmp1.times(2*(double)(Object)this.lambda_q);

//        System.out.println(tmp1.getRowDimension());
//        System.out.println(tmp1.getColumnDimension());
//        System.out.println(grad_tmp.getRowDimension());
//        System.out.println(grad_tmp.getColumnDimension());
        grad_tmp=grad_tmp.plus(tmp1);

        if((double)(Object)this.penalty_weight!=0.0)
        {
            if(!this.exact_penalty)
            {
                Matrix diff;
                if(one.get(1)==0.0)
                {
                    Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()+1));
                    Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                    diff=T_tmp_1.minus(T_tmp_2);
                }
                else if(one.get(1)==((double)this.num_attempts-1.0))
                {
                    Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                    Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()-1));
                    diff=T_tmp_1.minus(T_tmp_2);
                }
                else
                {
                    Matrix T_tmp_1=new Matrix(array_slice_3d(this.T, one.get(1).intValue()));
                    Matrix T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()-1));
                    diff=T_tmp_1.minus(T_tmp_2);
                    T_tmp_2=new Matrix(array_slice_3d(this.T, one.get(1).intValue()+1));
                    diff=diff.plus(T_tmp_2.minus(T_tmp_1));
                }
                double [][]diff_array=diff.getArrayCopy();
                for(int i=0;i<diff_array.length;i++){
                    for(int j=0;j<diff_array[0].length;j++){
                        if(diff_array[i][j]>0){
                            diff_array[i][j]=0;
                        }
                    }
                }
                Matrix val=new Matrix(dot_product(array_slice_2d(this.S,0,one.get(0).intValue()),diff_array));
//                System.out.println(val.getRowDimension());
//                System.out.println(val.getColumnDimension());
//                System.out.println(grad_tmp.getRowDimension());
//                System.out.println(grad_tmp.getColumnDimension());
                grad_tmp=grad_tmp.minus(val.transpose().times((-1)*(double)(Object)this.penalty_weight));
            }
        }
        grad=grad_tmp.getArrayCopy();
        return grad;
    }
    public double _grad_bias_s(List<Double> one,double obs,int obs_flag){
        double grad=0.0;
        if(obs_flag!=0){
            double pred=_get_prediction(one);
            if(this.binarized_question){
                grad-=2*(obs-pred)*pred*(1-pred);
            }
            else {
                grad-=2*(obs-pred);
            }
        }
        grad+=2*(double)(Object)this.lambda_bias*this.bias_s[one.get(0).intValue()];
        return grad;
    }
    public double _grad_bias_q(List<Double> one,double obs,int obs_flag){
        double grad=0.0;
        if(obs_flag!=0){
            double pred=_get_prediction(one);
            if(this.binarized_question){
                grad-=2*(obs-pred)*pred*(1-pred);
            }
            else {
                grad-=2*(obs-pred);
            }
        }
        grad+=2*(double)(Object)this.lambda_bias*this.bias_q[one.get(2).intValue()];
        return grad;
    }
    public double _grad_bias_a(List<Double> one,double obs,int obs_flag){
        double grad=0.0;
        if(obs_flag!=0){
            double pred=_get_prediction(one);
            if(this.binarized_question){
                grad-=2*(obs-pred)*pred*(1-pred);
            }
            else {
                grad-=2*(obs-pred);
            }
        }
        grad+=2*(double)(Object)this.lambda_bias*this.bias_a[one.get(1).intValue()];
        return grad;
    }

    public void _optimize_sgd(List<Double> one,double obs,int obs_flag){

        //optimize S
        double[][] grad_s=_grad_S_k(one,obs,obs_flag);
        for(int i=0;i<this.S[one.get(0).intValue()].length;i++){
            this.S[one.get(0).intValue()][i]-=((double)(Object)this.lr*grad_s[0][i]);
        }
        if((double)(Object)this.lambda_s==0.0){
            double sum=0.0;
            for(int i=0;i<this.S[one.get(0).intValue()].length;i++){
                if(this.S[one.get(0).intValue()][i]<0.0){
                    this.S[one.get(0).intValue()][i]=0;
                }
                sum+=this.S[one.get(0).intValue()][i];
            }
            if(sum!=0.0){
                for(int i=0;i<this.S[one.get(0).intValue()].length;i++){
                        this.S[one.get(0).intValue()][i]/=sum;
                }
            }
        }

        //optimize Q
        double[][] grad_q=_grad_Q_k(one,obs,obs_flag);
//        System.out.println(grad_q.length);
//        System.out.println(this.Q.length);
        for(int i=0;i<this.Q.length;i++){
            this.Q[i][one.get(2).intValue()]-=((double)(Object)this.lr*grad_q[i][0]);
        }
        double sum=0.0;
        for(int i=0;i<this.Q.length;i++)
        {
            if(this.Q[i][one.get(2).intValue()]<0.0)
            {
                this.Q[i][one.get(2).intValue()]=0;
            }
            sum+=this.Q[i][one.get(2).intValue()];
        }
        if((double)(Object)this.lambda_q==0){
            if(sum!=0.0){
                for(int i=0;i<this.Q.length;i++){
                    this.Q[i][one.get(2).intValue()]/=sum;
                }
            }

        }

        //the updated Q will be used for computing gradient of T
        double[][] grad_t=_grad_T_ij(one,obs,obs_flag);
        for(int i=0;i<this.T.length;i++){
            for(int j=0;j<this.T[i][one.get(1).intValue()].length;j++){
                this.T[i][one.get(1).intValue()][j]-=((double)(Object)this.lr*grad_t[i][j]);
            }
        }

        //train the bias(es)
        this.bias_s[one.get(0).intValue()]-=(double)(Object)this.lr*_grad_bias_s(one,obs,obs_flag);
        this.bias_a[one.get(1).intValue()]-=(double)(Object)this.lr*_grad_bias_a(one,obs,obs_flag);
        this.bias_q[one.get(2).intValue()]-=(double)(Object)this.lr*_grad_bias_q(one,obs,obs_flag);

    }

     public  void training(){
        boolean converge=false;

         double []result_loss=_get_loss();
         double loss=result_loss[0];double rmse=result_loss[1];double reg_loss=result_loss[2];double penalty=result_loss[3];
         double bias_reg=result_loss[4];
         this.loss_list=new ArrayList<Double>();
         this.loss_list.add(loss);

        double iter=0;
        while(!converge)
        {
            Collections.shuffle(this.train_order,this.random);

            double[][]best_S=new double[this.S.length][this.S[0].length];
            for(int i=0;i<this.S.length;i++){
                best_S[i]=this.S[i].clone();
            }

            double[][][]best_T=new double[this.T.length][this.T[0].length][this.T[0][0].length];
            for(int i=0;i<this.T.length;i++){
                for(int j=0;j<this.T[0].length;j++){
                    best_T[i][j]=this.T[i][j].clone();
                }
            }

            double[][]best_Q=new double[this.Q.length][this.Q[0].length];
            for(int i=0;i<this.Q.length;i++){
                best_Q[i]=this.Q[i].clone();
            }

            double []best_bias_s=this.bias_s.clone();
            double []best_bias_q=this.bias_q.clone();
            double []best_bias_a=this.bias_a.clone();

            for(int i=0;i<this.train_order.size();i++){
                List<Double> one=this.train_order.get(i).subList(0,3);
                double obs=this.obs_tensor[one.get(0).intValue()][one.get(1).intValue()][one.get(2).intValue()];
                if (obs==0.0)
                {
                        _optimize_sgd(one,obs,0);
                }
                else{
                        _optimize_sgd(one,obs,1);
                }
            }

            print_matrix(this.S);
            System.exit(0);
            result_loss=_get_loss();
            loss=result_loss[0];rmse=result_loss[1];reg_loss=result_loss[2];penalty=result_loss[3];
            bias_reg=result_loss[4];
            //System.out.println(loss);
            // # checking the stopping criteria
            double mean_loss=0;
            int count=0;
            while(count<10 && count+1<=loss_list.size()){
                mean_loss+=loss_list.get(loss_list.size()-count-1);
                count++;
            }
            mean_loss/=(double)count;
            if(Math.abs(loss-this.loss_list.get(this.loss_list.size()-1))<(double)(Object)this.tol || iter==(double)(Object)this.max_iter){

                this.loss_list.add(loss);
                converge=true;
                this.S=best_S;
                this.T=best_T;
                this.Q=best_Q;
                this.bias_s=best_bias_s;
                this.bias_a=best_bias_a;
                this.bias_q=best_bias_q;
            }
            else if(iter>10.0 && loss>=mean_loss)
            {
                converge=true;
                this.S=best_S;
                this.T=best_T;
                this.Q=best_Q;
                this.bias_s=best_bias_s;
                this.bias_a=best_bias_a;
                this.bias_q=best_bias_q;
            }
            else if(loss>this.loss_list.get(this.loss_list.size()-1))
            {
                this.loss_list.add(loss);
                this.lr*=0.9;
            }
            else{
                this.loss_list.add(loss);
                iter++;
            }
        }
     }

     public double[] testing(List<List<Double>> test_data)
     {
         double q_count=0;double square_error=0;double abs_error=0;double percent_error=0;double abs_percent_error=0;
         for(int i=0;i<test_data.size();i++)
         {
             double obs=test_data.get(i).get(3);
             q_count++;
             List<Double> one=test_data.get(i).subList(0,3);
             double pred=_get_prediction(one);
             square_error+=Math.pow(obs-pred,2);
             abs_error+=Math.abs(obs-pred);
             if(obs==0.0){
                obs=1e-5;
             }
             percent_error+=(obs-pred)/obs;
             abs_percent_error+=Math.abs(obs-pred)/obs;

         }
         double []result;
         if(q_count==0){
             result= new double[]{0, 0, 0, 0, 0};
             return  result;
         }
         else{
             double rmse=Math.sqrt(square_error / q_count);
             double mae=abs_error / q_count;
             double mpe=percent_error / q_count;
             double mape=abs_percent_error / q_count;
             result= new double[]{q_count,rmse,mae,mpe,mape};
             return  result;
         }
     }
    public static void print_matrix(double [][]data){
        for(int i=0;i<data.length;i++){
            for(int j=0;j<data[i].length;j++){
                System.out.print(data[i][j]+" ");
            }
            System.out.println();

        }
    }



}
