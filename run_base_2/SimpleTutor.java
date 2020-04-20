package run_base2;

import Jama.Matrix;
import lab2020_recom.Config_re;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.*;

public class SimpleTutor<T> {
//    int seed=1;
    Random random = new Random();
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
    public double penalty_weight;
    public double markovian_steps;
    public double lr;
    public T tol;
    public T max_iter;

    public JSONObject test_users;
    public JSONObject users_data;
    public JSONObject test_users_logged_perf;
    public JSONObject next_questions_dict;
    public JSONObject question_score_dict;

    public boolean binarized_question = true;
    public double current_test_attempt;
    public List<Double> loss_list;
    public List<List<Double>>  val_data;
    public List<List<Double>> train_data_markovian;
    public List<Double> target_rewards;
    public List<Double> logged_rewards;
    public JSONObject test_users_historical_records;

    public double[][] S ;
    public double[][][] T ;
    public double[][] Q ;

    public double[]bias_s;
    public double[]bias_q;
    public double[]bias_a;

    public JSONObject current_questions;
    public JSONObject current_states;
    public JSONObject current_scores;
    public JSONObject test_users_perf;
    public JSONObject rare_questions;

    public double top_k;
    public JSONObject test_users_records;
    public JSONObject test_users_logged_testing_perf;
    public JSONObject test_users_logged_before_testing_perf;


    public SimpleTutor(Config config){
        this.train_data=config.train;
        this.num_users=((Long)(Object)config.num_users).intValue();
        this.num_attempts=((Long)(Object)config.num_attempts).intValue();
        this.num_questions=((Long)(Object)config.num_questions).intValue();
        this.num_skills=((Double)(Object)config.num_skills).intValue();
        this.num_concepts=((Double)(Object)config.num_concepts).intValue();
        //this.num_examples=((Long)(Object)config.num_discussions).intValue();
        this.lambda_s=(T)config.lambda_s;
        this.lambda_t=(T)config.lambda_t;
        this.lambda_q=(T)config.lambda_q;
        this.lambda_bias=(T)config.lambda_bias;
        this.penalty_weight=(double)config.penalty_weight;
        this.markovian_steps=(double)config.markovian_steps;
        if(this.markovian_steps==0){
            this.penalty_weight=0;
        }
        if(this.penalty_weight==0){
            this.markovian_steps=0;
        }

        this.top_k=(double)config.top_k;
        this.lr=(double)config.lr;
        this.tol=(T)config.tol;
        this.max_iter=(T)config.max_iter;

        this.test_users = config.test_users;

        this.users_data = config.users_data;
        this.question_score_dict = config.question_score_dict;

        this.rare_questions=new JSONObject();
        Iterator tmp1=this.question_score_dict.keySet().iterator();
        for(int i=0;i<this.question_score_dict.keySet().size();i++)
        {
            String tmp_key=(String)tmp1.next();

            JSONArray temp=(JSONArray) question_score_dict.get(tmp_key);
            System.out.println(String.format("%s, %d",tmp_key,temp.size()));

            if (temp.size()<5){
                this.rare_questions.put(Double.parseDouble(tmp_key),true);
            }
        }

        this.test_users_logged_testing_perf = config.test_users_logged_testing_perf;

        this.test_users_logged_before_testing_perf = config.test_users_logged_before_testing_perf;

        this.test_users_records = config.test_users_records;
        this.next_questions_dict = config.next_questions_dict;

        this.current_test_attempt=config.start_test_attempt;




        JSONObject train_data_dict=new JSONObject();
        JSONObject train_data_markovian_dict=new JSONObject();
        this.train_data_markovian=new ArrayList<List<Double>>();

        for(int i=0; i< this.train_data.size() ; i++)
        {
            String key=String.format("%f%f%f",this.train_data.get(i).get(0),this.train_data.get(i).get(1),
                    this.train_data.get(i).get(2));

            if (train_data_dict.get(key)==null){
                train_data_dict.put(key,(double)train_data.get(i).get(3));
            }

        }

        for(int i=0; i< this.train_data.size() ; i++)
        {
            double range=Math.min(this.num_attempts,
                    this.train_data.get(i).get(1)+(double)(Object)this.markovian_steps+1);

            for(double j=(double)this.train_data.get(i).get(1)+1;j<range;j++) {
                String key=String.format("%f%f%f",(double)train_data.get(i).get(0),j,
                        (double)train_data.get(i).get(2));
                if (train_data_dict.get(key)==null){
                    if(train_data_markovian_dict.get(key)==null){
                        train_data_markovian_dict.put(key,true);
                        List<Double> temp=new ArrayList<Double>();
                        temp.add(train_data.get(i).get(0));temp.add(j);temp.add(train_data.get(i).get(2));
                        this.train_data_markovian.add(temp);
                    }
                }

            }
        }



       this.next_questions_dict =  config.next_questions_dict;

        this.S=new double[this.num_users][this.num_skills];
        two_d_matrix(this.S,this.num_users,this.num_skills);
        this.T=new double[this.num_skills][this.num_attempts][this.num_concepts];
        three_d_matrix(this.T,this.num_skills,this.num_attempts,this.num_concepts);
        this.Q=new double[this.num_concepts][this.num_questions];
        two_d_matrix(this.Q,this.num_concepts,this.num_questions);

        this.bias_s=new double[this.num_users];
        this.bias_q=new double[this.num_questions] ;
        this.bias_a=new double[this.num_attempts];



        JSONObject records=new JSONObject();
        JSONObject state_records=new JSONObject();
        for(int i=0; i< this.train_data.size() ; i++)
        {
            double student=LongOrDoubleToDouble(train_data.get(i).get(0));
            double attempt=LongOrDoubleToDouble(train_data.get(i).get(1));
            double question=LongOrDoubleToDouble(train_data.get(i).get(2));
            double obs=LongOrDoubleToDouble(train_data.get(i).get(3));

            if(records.get(student)==null){
                List<Double> temp=new ArrayList<Double>();
                temp.add(train_data.get(i).get(1));temp.add(train_data.get(i).get(2));temp.add(train_data.get(i).get(3));
                records.put(student,temp);
            }
            else{
                List<Double> one=(List<Double>)records.get(student);

                double max_attempt=one.get(0);
                if(attempt>max_attempt){
                    List<Double> temp=new ArrayList<Double>();
                    temp.add(attempt);temp.add(question);temp.add(obs);
                    records.put(student,temp);
                }
            }
        }

//        JSONArray test=(JSONArray) this.users_data.get("0");
//        System.out.println(test.get(0).getClass());
//        System.exit(0);
        this.test_users_historical_records=new JSONObject();
        Iterator tmp3=this.users_data.keySet().iterator();
        while(tmp3.hasNext()){
            double user_key=Double.parseDouble((String)tmp3.next());
            JSONArray all_records=(JSONArray )this.users_data.get(Integer.toString((int)user_key));

//            System.out.println(this.test_users.keySet());
//            System.out.println(user_key);
            if(this.test_users.get(Integer.toString((int)user_key))!=null){
              //  System.out.println("dsdsdsdsds");

                if(this.test_users_historical_records.get(user_key)==null){
                    this.test_users_historical_records.put(user_key,new JSONObject());
                }

                for (int i=0;i<all_records.size();i++){
                    JSONArray one=(JSONArray) all_records.get(i);


                    double att=LongOrDoubleToDouble(one.get(1));

//                    System.out.println(att);
//                    System.out.println(this.current_test_attempt);
                    if(att<this.current_test_attempt){


                        JSONObject two=(JSONObject) this.test_users_historical_records.get(user_key);
                        double ques=LongOrDoubleToDouble(one.get(2));
                        double obs=LongOrDoubleToDouble(one.get(3));
                        //System.out.println(String.format("%f,%f,%f",user_key,ques,obs));
                        JSONArray three;
                        if(two.get(ques)==null){
                            three=new JSONArray();
                            two.put(ques,three);
                        }
                        else
                            {
                                three= (JSONArray) two.get(ques);
                        }
                        //System.out.println(one.get(2).getClass());


                        //System.out.println(three.size());
                        three.add(obs);

                    }

                }

            }
        }
//        System.out.println(test_users_historical_records);
//        System.exit(0);








//        System.out.println(this.test_users.keySet());
//        System.out.println(records.keySet());
        this.current_scores=new JSONObject();
        this.current_questions=new JSONObject();
        this.test_users_perf=new JSONObject();
        Iterator tmp=this.test_users.keySet().iterator();
        for(int i=0;i<this.test_users.keySet().size();i++)
        {
            double key=Double.parseDouble((String)tmp.next());
            List<Double> temp=( List<Double>)records.get(key);
            this.current_questions.put(key,temp.get(1));
            this.current_scores.put(key,temp.get(2));
        }

        Iterator tmp2=this.test_users.keySet().iterator();
        for(int i=0;i<this.test_users.keySet().size();i++)
        {
            double key=Double.parseDouble((String)tmp2.next());
            List<Double> one=new ArrayList<Double>();
            this.test_users_perf.put(key,one);
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
    public  double sigmoid(double x,int derivative_flag){
        double sigm;
        if(x>100){
            sigm=1;
        }
        else if(x<-100){
            sigm=0;
        }
        else {
            sigm = 1 / (1 + Math.exp(-x));
        }

        if (derivative_flag==1) {
            return sigm * (1 - sigm);
        }
        return sigm;
    }
    public double long_to_double(Long data){
        double result;
        Long tmp_num = data;
        result = tmp_num.doubleValue();
        return result;
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

    //[:,n] or [n,:]
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

    // [:,n,:]
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
    public  double _get_question_prediction(List<Double> one){

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

        if (this.binarized_question){
            pred = sigmoid(pred,0);
        }
        return pred;
    }
    public  double _get_penalty(){
        double penalty = 0.0;
        for(int i=0;i<this.train_data.size();i++)
        {
            List<Double> one=new ArrayList<>(this.train_data.get(i).subList(0,4));
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


                diff=dot_product(
                        dot_product(array_slice_2d(this.S,0,one.get(0).intValue()),tmp)
                        ,array_slice_2d(this.Q,1,one.get(2).intValue()))[0][0];
                penalty-=diff;

            }
        }

        for(int i=0;i<this.train_data_markovian.size();i++)
        {
            List<Double> one = new ArrayList<>(this.train_data_markovian.get(i).subList(0, 3));
            if (one.get(1) >= 1)
            {
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


                diff=dot_product(
                        dot_product(array_slice_2d(this.S,0,one.get(0).intValue()),tmp)
                        ,array_slice_2d(this.Q,1,one.get(2).intValue()))[0][0];
                penalty-=diff;
            }
        }



        return penalty;
    }
    public  double[] _get_loss(){
        double loss=0.0;double square_loss = 0.0;double bias_reg = 0.0;
        for(int i=0;i<this.train_data.size();i++)
        {
            double pred=_get_question_prediction(this.train_data.get(i).subList(0,3));
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

        loss+=bias_reg+penalty+(double)(Object)this.markovian_steps;
        double result[]={loss,rmse,reg_loss,penalty,bias_reg};
        return result;
    }

    public double[][] _grad_S_k(List<Double> one,double obs,int obs_flag){
        double grad[][]=new double[1][this.S[one.get(0).intValue()].length];
        //two_d_matrix(grad,1,this.S[one.get(0).intValue()].length);
        Matrix grad_tmp=new Matrix(grad);

        if(obs_flag!=0)
        {
            double pred=_get_question_prediction(one);
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


        grad=grad_tmp.getArrayCopy();
        return grad;
    }
    public double[][] _grad_T_ij(List<Double> one,double obs,int obs_flag){
        double grad[][]=new double[this.T.length] [this.T[0][one.get(2).intValue()].length];
        Matrix grad_tmp=new Matrix(grad);
        if(obs_flag!=0)
        {
            double pred=_get_question_prediction(one);
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



            diff_array=dot_product(
                    dot_product(array_slice_2d(this.S,0,one.get(0).intValue()), diff_array),
                    array_slice_2d(this.Q,1,one.get(2).intValue()));
            tmp=new Matrix(outer_product(
                    array_slice_2d(this.S,0,one.get(0).intValue()),
                    array_slice_2d(this.Q,1,one.get(2).intValue())));
            grad_tmp=grad_tmp.minus(tmp.times((double)(Object)this.penalty_weight*diff_array[0][0]));

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


            diff_array=dot_product(
                    dot_product(array_slice_2d(this.S,0,one.get(0).intValue()), diff_array),
                    array_slice_2d(this.Q,1,one.get(2).intValue()));
            tmp=new Matrix(outer_product(
                    array_slice_2d(this.S,0,one.get(0).intValue()),
                    array_slice_2d(this.Q,1,one.get(2).intValue())));
            grad_tmp=grad_tmp.minus(tmp.times((-1)*(double)(Object)this.penalty_weight*diff_array[0][0]));

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


            diff_array=dot_product(
                    dot_product(array_slice_2d(this.S,0,one.get(0).intValue()), diff_array),
                    array_slice_2d(this.Q,1,one.get(2).intValue()));
            tmp=new Matrix(outer_product(
                    array_slice_2d(this.S,0,one.get(0).intValue()),
                    array_slice_2d(this.Q,1,one.get(2).intValue())));
            grad_tmp=grad_tmp.minus(tmp.times((-1)*(double)(Object)this.penalty_weight*diff_array[0][0]));


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


            diff_array=dot_product(
                    dot_product(array_slice_2d(this.S,0,one.get(0).intValue()), diff_array),
                    array_slice_2d(this.Q,1,one.get(2).intValue()));
            tmp=new Matrix(outer_product(
                    array_slice_2d(this.S,0,one.get(0).intValue()),
                    array_slice_2d(this.Q,1,one.get(2).intValue())));
            grad_tmp=grad_tmp.minus(tmp.times((double)(Object)this.penalty_weight*diff_array[0][0]));



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
            double pred=_get_question_prediction(one);
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


        grad=grad_tmp.getArrayCopy();
        return grad;
    }
    public double _grad_bias_s(List<Double> one,double obs,int obs_flag){
        double grad=0.0;
        if(obs_flag!=0){
            double pred=_get_question_prediction(one);
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
            double pred=_get_question_prediction(one);
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
            double pred=_get_question_prediction(one);
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
        if((double)(Object)this.lambda_s==0.0)
        {
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
        else{
            for(int i=0;i<this.S[0].length;i++)
            {
                if(this.S[one.get(0).intValue()][i]<0.0)
                {
                    this.S[one.get(0).intValue()][i]=0;
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

    //初始化
    public void set_train_data_markovian(){
        JSONObject train_data_dict=new JSONObject();
        JSONObject train_data_markovian_dict=new JSONObject();
        this.train_data_markovian=new ArrayList<List<Double>>();

        for(int i=0; i< this.train_data.size() ; i++)
        {
            String key=String.format("%f%f%f",this.train_data.get(i).get(0),this.train_data.get(i).get(1),
                    this.train_data.get(i).get(2));

            if (train_data_dict.get(key)==null){
                train_data_dict.put(key,(double)train_data.get(i).get(3));
            }

        }

        for(int i=0; i< this.train_data.size() ; i++)
        {
            double range=Math.min(this.num_attempts,
                    this.train_data.get(i).get(1)+(double)(Object)this.markovian_steps+1);

            for(double j=(double)this.train_data.get(i).get(1)+1;j<range;j++) {
                String key=String.format("%f%f%f",(double)train_data.get(i).get(0),j,
                        (double)train_data.get(i).get(2));
                if (train_data_dict.get(key)==null){
                    if(train_data_markovian_dict.get(key)==null){
                        train_data_markovian_dict.put(key,true);
                        List<Double> temp=new ArrayList<Double>();
                        temp.add(train_data.get(i).get(0));temp.add(j);temp.add(train_data.get(i).get(2));
                        this.train_data_markovian.add(temp);
                    }
                }

            }
        }
    }

    //初始化
    public void set_current_questions_states_scores_perf(){
        JSONObject records=new JSONObject();
        JSONObject state_records=new JSONObject();
        for(int i=0; i< this.train_data.size() ; i++)
        {
            if(records.get(train_data.get(i).get(0))==null){
                List<Double> temp=new ArrayList<Double>();
                temp.add(train_data.get(i).get(1));temp.add(train_data.get(i).get(2));temp.add(train_data.get(i).get(3));
                records.put(train_data.get(i).get(0),temp);
            }
            else{
                List<Double> one=(List<Double>)records.get(train_data.get(i).get(0));

                double max_attempt=one.get(0);
                if(train_data.get(i).get(1)>max_attempt){
                    List<Double> temp=new ArrayList<Double>();
                    temp.add(train_data.get(i).get(1));temp.add(train_data.get(i).get(2));temp.add(train_data.get(i).get(3));
                    records.put(train_data.get(i).get(0),temp);
                }
            }
        }

//        System.out.println(this.test_users.keySet());
//        System.out.println(records.keySet());
        this.current_scores=new JSONObject();
        this.current_questions=new JSONObject();
        this.test_users_perf=new JSONObject();
        Iterator tmp=this.test_users.keySet().iterator();
        for(int i=0;i<this.test_users.keySet().size();i++)
        {
            double key=(double)tmp.next();
            List<Double> temp=( List<Double>)records.get(key);
            this.current_questions.put(key,temp.get(1));
            this.current_scores.put(key,temp.get(2));
        }

        Iterator tmp1=this.test_users.keySet().iterator();
        for(int i=0;i<this.test_users.keySet().size();i++)
        {
            double key=(double)tmp1.next();
            List<Double> one=new ArrayList<Double>();
            this.test_users_perf.put(key,one);
        }

    }

    public void training(){

        this.val_data=new ArrayList<List<Double>>();
        Collections.shuffle(this.train_data,this.random);

        for(int i=0; i<this.train_data.size()*0.1; i++)
        {
            List<Double> temp=this.train_data.get(i);

            //System.out.println(i);
            if(this.test_users.get(Integer.toString(temp.get(0).intValue()))==null && temp.get(1)>0)
            {
                //print_arraylist(temp);
                this.val_data.add(temp);
               // System.out.println(i);
            }
        }

        for(int i=0; i<this.train_data.size(); i++)
        {
            List<Double> temp=this.train_data.get(i);
            if(this.test_users.get(Integer.toString(temp.get(0).intValue()))!=null && temp.get(1)<this.current_test_attempt && temp.get(1)>0)
            {
                double prob=random.nextInt(10)*0.1;
                if(prob<0.2){
                    this.val_data.add(temp);
                }
            }

        }

        System.out.println("==========================================");
        System.out.println("test attempt: "+this.current_test_attempt+", validation data size:"+this.val_data.size());
        for(int i=0; i< this.val_data.size() ; i++){
            List<Double> temp=this.val_data.get(i);
            this.train_data.remove(temp);
        }


        double startTime = System.currentTimeMillis();

        boolean converge=false;

        double []result_loss=_get_loss();
        double loss=result_loss[0];double rmse=result_loss[1];double reg_loss=result_loss[2];double penalty=result_loss[3];
        double bias_reg=result_loss[4];
        this.loss_list=new ArrayList<Double>();
        this.loss_list.add(loss);

        double iter=0;
        double val_q_count = 0;
        List<Double> val_q_rmse_list=new ArrayList<>();
        val_q_rmse_list.add(1.0);

        System.out.println("******************"+"[ Training Results ]"+"*******************");
        double[][]best_S=new double[this.S.length][this.S[0].length];
        double[][][]best_T=new double[this.T.length][this.T[0].length][this.T[0][0].length];
        double[][]best_Q=new double[this.Q.length][this.Q[0].length];
        double []best_bias_s=this.bias_s.clone();
        double []best_bias_q=this.bias_q.clone();
        double []best_bias_a=this.bias_a.clone();
        while(!converge) {

            double  runTime = System.currentTimeMillis() - startTime;

//            System.out.println(String.format("iter: %f, lr: %.4f, loss: %.2f, rmse: %.5f reg_T: %.2f, penalty: %.5f, bias_reg: %.3f, run time: %.2f",
//                    iter,lr,loss,rmse,reg_loss,penalty, bias_reg,runTime));
//            System.out.println(String.format("validation: val_q_count: %f, val_q_rmse: %.5f",val_q_count, val_q_rmse_list.get(val_q_rmse_list.size()-1)));

            Collections.shuffle(this.train_data,this.random);
            Collections.shuffle(this.train_data_markovian,this.random);

            for(int i=0;i<this.S.length;i++){
                best_S[i]=this.S[i].clone();
            }

            for(int i=0;i<this.T.length;i++){
                for(int j=0;j<this.T[0].length;j++){
                    best_T[i][j]=this.T[i][j].clone();
                }
            }

            for(int i=0;i<this.Q.length;i++){
                best_Q[i]=this.Q[i].clone();
            }

            best_bias_s=this.bias_s.clone();
            best_bias_q=this.bias_q.clone();
            best_bias_a=this.bias_a.clone();

            for(int i=0;i<this.train_data.size();i++){
                List<Double> one=this.train_data.get(i).subList(0,3);
                _optimize_sgd(one,this.train_data.get(i).get(3),1);
            }

            for(int i=0;i<this.train_data_markovian.size();i++){
                List<Double> one=this.train_data_markovian.get(i).subList(0,3);
                _optimize_sgd(one,0,0);
            }

            result_loss=_get_loss();
            loss=result_loss[0];rmse=result_loss[1];reg_loss=result_loss[2];penalty=result_loss[3];
            bias_reg=result_loss[4];
            double []result_validate=validate(this.val_data);
            val_q_count=result_validate[0];
            double val_q_rmse=result_validate[1];

            double mean_loss=0;
            double mean_val_q_rmse=0;
            int count=0;
            while(count<3 && count+1<=loss_list.size()){
                mean_loss+=loss_list.get(loss_list.size()-count-1);
                count++;
            }
            mean_loss/=(double)count;
            count=0;
            while(count<3 && count+1<=val_q_rmse_list.size()){
                mean_val_q_rmse+=val_q_rmse_list.get(val_q_rmse_list.size()-count-1);
                count++;
            }
            mean_val_q_rmse/=(double)count;


           // System.out.println((double)(Object)this.tol);
            if(iter==(double)(Object)this.max_iter){
                this.loss_list.add(loss);
                converge=true;
            }
            else if(Math.abs(val_q_rmse - val_q_rmse_list.get(val_q_rmse_list.size()-1)) <  (double)(Object)this.tol  ){
                this.loss_list.add(loss);
                converge=true;
            }
            else if(iter>5 && val_q_rmse>=mean_val_q_rmse){
                converge=true;
            }
            else if(iter>5 && loss>=mean_loss){
                converge=true;
            }
            else if(val_q_rmse >= val_q_rmse_list.get(val_q_rmse_list.size()-1)){
                this.loss_list.add(loss);
                val_q_rmse_list.add(val_q_rmse);
                iter+=1;
                this.lr *= 0.5;
            }
            else if(loss==0){
                this.lr *= 0.1;
            }
            else{
                this.loss_list.add(loss);
                val_q_rmse_list.add(val_q_rmse);
                iter += 1;
            }


        }
        this.S=best_S;
        this.T=best_T;
        this.Q=best_Q;
        this.bias_s=best_bias_s;
        this.bias_a=best_bias_a;
        this.bias_q=best_bias_q;
        for(int i=0; i< this.val_data.size() ; i++){
            List<Double> record=this.val_data.get(i);
            this.train_data.remove(record);
        }
        System.out.println("-----------------------------");
    }

    public double[] validate( List<List<Double>> val_data){
        double q_count=0;double square_error =0;double penalty=0;

        for(int i=0;i<val_data.size();i++){
            List<Double> one=new ArrayList<Double>(val_data.get(i).subList(0,3));
            q_count+=1;

            double pred=_get_question_prediction(one);


            double tmp=one.get(1);
            one.set(1,tmp-1);
//            System.out.println(one.get(1));
//            System.out.println(val_data.get(i).get(1));
//            System.exit(0);
            //System.out.println(one.get(1));
            penalty+=pred-_get_question_prediction(one);
            square_error+=Math.pow((val_data.get(i).get(3)-pred),2);

        }
        if(q_count==0){
            double result[]={0,0};
            return result;
        }
        else {
            double rmse=Math.sqrt(square_error/q_count);
            double result[]={q_count,rmse};
            return result;
        }

    }

    //1*n n*1
    public double cosine( double [][]a,  double [][]b){

        double dot=dot_product(a,b)[0][0];

        double norm_a=0;
        for(double tmp:a[0]){
            norm_a+=tmp*tmp;
        }
        norm_a=Math.sqrt(norm_a);
        double norm_b=0;
        for(double[] tmp:b){
            norm_b+=tmp[0]*tmp[0];
        }
        norm_b=Math.sqrt(norm_b);
        double result=1-(dot/(norm_a*norm_b));
        return result;
    }


    public void generate_next_items()
    {
        System.out.println("******************"+"[Recommendation Results]"+"*******************");
        double  hit_count = 0;
        double miss_count = 0;

        Iterator tmp=this.test_users.keySet().iterator();
        for(int i=0;i<this.test_users.keySet().size();i++) {
            double student = Double.parseDouble( (String)tmp.next());
            double current_q = (double) this.current_questions.get(student);
            double current_score = (double) this.current_scores.get(student);
//            System.out.println("student, cur-q:");
//            System.out.println(student);
//            System.out.println(  current_q);
//            System.out.println(current_score);
            List<Double> updated_proximity_list = new ArrayList<Double>();
            double current_difficulty = 0;
            double current_est_score = 0;


            for (double question = 0; question < this.num_questions; question++) {
                double temp = (double) (Object) this.current_test_attempt - 1;
                double[][] pred_matrix = dot_product(
                        dot_product(array_slice_2d(this.S, 0, (int) student),
                                array_slice_3d(this.T, (int) temp)
                        ), array_slice_2d(this.Q, 1, (int) question));

                double pred = pred_matrix[0][0];
                double est_score = sigmoid(pred + this.bias_q[(int) question] + this.bias_s[(int) student] +
                                this.bias_a[(int) (this.current_test_attempt - 1)]
                        , 0);


                //temp = (double) (Object) this.current_test_attempt - 1;
                double[][] knowledge = dot_product(array_slice_2d(this.S, 0, (int) student),
                        array_slice_3d(this.T, (int) temp));

                for (int s = 0; s < knowledge.length; s++) {
                    for (int t = 0; t < knowledge[0].length; t++) {
                        knowledge[s][t] += this.bias_s[(int) student];
                    }
                }
                double[][] tmp_Q = array_slice_2d(this.Q, 1, (int) question);

                double gap = cosine(knowledge, array_slice_2d(this.Q, 1, (int) question));// 1*n, m*1

//                System.out.println("question,cos:");
//                System.out.println(question);
//                System.out.println(gap);
//                System.out.println(est_score);
                double proximity = (1 - gap) * (1.0 - est_score);

                if (current_q == question) {
                    current_difficulty = proximity; //record the difficulty of current question
                    current_est_score = est_score;
                }
                updated_proximity_list.add(proximity);


            }


//            for(double tmp1:updated_difficulties){
//                System.out.print(tmp1+" ");
//            }
            List<Double> sorted_proximity_list = new ArrayList<Double>(updated_proximity_list);
            Collections.sort(sorted_proximity_list, Collections.reverseOrder());

//            print_arraylist(updated_proximity_list);
//            System.out.println();
//            print_arraylist(sorted_proximity_list);


            List<Double>candidates=new ArrayList<Double>();
            List<List<Double>>candidate_and_proximity=new ArrayList<List<Double>>();
//
//
            List<Double>one=(List<Double>)this.question_score_dict.get(String.format("%d",(int)current_q));
            double mean_score_current_question=0;
            for(double tmp1:one){
                mean_score_current_question+=tmp1;
            }
            mean_score_current_question/=one.size();


            for(double prox: sorted_proximity_list)
            {

                if(candidates.size()<this.top_k)
                {
                    double next_question=updated_proximity_list.indexOf(prox);

                    JSONObject s_tmp=(JSONObject) this.test_users_historical_records.get(student);
                    JSONArray q_tmp=(JSONArray) s_tmp.get(next_question);
//                    System.out.println(s_tmp.keySet());
//                    System.out.println(q_tmp.get(0));
                    if(this.rare_questions.get(next_question)==null)
                    {
                        if(s_tmp.get(next_question)==null)
                        {
                            candidates.add(next_question);
                            List<Double> two=new ArrayList<Double>();
                            two.add(next_question);two.add(prox);
                            candidate_and_proximity.add(two);
                        }

                        else if((double)q_tmp.get(q_tmp.size()-1)!=1.0)
                        {
                           // System.out.println(String.format("q_tmp size: %d",q_tmp.size()));
                            candidates.add(next_question);
                            List<Double> two=new ArrayList<Double>();
                            two.add(next_question);two.add(prox);
                            candidate_and_proximity.add(two);
                        }
                    }

                }


            }

            //System.out.println(this.test_users_records.keySet());
            JSONObject test_users_records_s=(JSONObject) this.test_users_records.get(student);

            if(test_users_records_s.get(this.current_test_attempt)==null){
                continue;
            }

            double true_next_question=(double)test_users_records_s.get(this.current_test_attempt);
//            System.out.println(true_next_question);
//            System.out.println("........");

//            System.out.println(String.format("student %.0f",student));
//            System.out.print("sorted proximity:  ");
//            print_arraylist(sorted_proximity_list);


            if(candidates.contains(true_next_question))
            {
                System.out.println(String.format("HIT student %.0f", student));
                System.out.println(String.format("current question and score: (%f, %.3f), and estimated score: %.3f", current_q, current_score, current_est_score));
                System.out.print("candidates: ");
                print_arraylist(candidates);
                System.out.println();
                System.out.print("historical: ");
                System.out.println(this.test_users_historical_records.get(student));
                System.out.println(String.format("true next question: %.0f and proximity: %f", true_next_question,
                        updated_proximity_list.get((int) true_next_question)));
                System.out.println(String.format("true next question distribution (next-q: count): %s",
                        this.next_questions_dict.get(Integer.toString((int) current_q))));
                hit_count ++;

                double ndch = 1.0 / (Math.log(candidates.indexOf(true_next_question) + 1 + 1) / Math.log(2));
               // System.out.println(candidates.indexOf(true_next_question));
                double total_count = 0;

                JSONObject current_tmp = (JSONObject) this.next_questions_dict.get(Integer.toString((int) current_q));
                Iterator question_i = current_tmp.keySet().iterator();

                while (question_i.hasNext()) {
                    String question = (String) question_i.next();
                    total_count += long_to_double((long) current_tmp.get(question));

                }
                double propensity_score = long_to_double((long) current_tmp.get(Integer.toString((int) true_next_question))) / total_count;
                double propensity_ndch = ndch / propensity_score;

//                System.out.println(String.format("ndch score: %f , propensity score: %f, propensity ndch score: %f", ndch, propensity_score, propensity_ndch));
//
//                System.out.println("---------------------------");

                List<Double> test_s = (List<Double>) this.test_users_perf.get(student);
                test_s.add(propensity_ndch);

            }

            else{
                System.out.println(String.format("MISS student %.0f", student));
                System.out.println(String.format("current question and score: (%f, %.3f), and estimated score: %.3f", current_q, current_score, current_est_score));
                System.out.print("candidates: ");
                print_arraylist(candidates);
                System.out.println();
                System.out.print("historical: ");
                System.out.println(this.test_users_historical_records.get(student));
                System.out.println(String.format("true next question: %.0f and proximity: %f", true_next_question,
                        updated_proximity_list.get((int) true_next_question)));
                System.out.println(String.format("true next question distribution (next-q: count): %s",
                        this.next_questions_dict.get(Integer.toString((int) current_q))));
                System.out.println("---------------------------");
                miss_count++;

                List<Double> test_s = (List<Double>) this.test_users_perf.get(student);
                test_s.add(0.0);

            }

            JSONObject s_tmp=(JSONObject) this.test_users_historical_records.get(student);
            JSONArray q_tmp;
            if(s_tmp.get(current_q)==null){
                q_tmp=new JSONArray();
                s_tmp.put(current_q,q_tmp);
            }
            q_tmp=(JSONArray) s_tmp.get(current_q);
            q_tmp.add(current_score);
            this.current_questions.put(student,true_next_question);
            JSONArray user_tmp=(JSONArray) this.users_data.get(Integer.toString((int)student));
            JSONArray user_att_tmp=(JSONArray)user_tmp.get((int)this.current_test_attempt);
            this.current_scores.put(student,user_att_tmp.get(3));

            //System.exit(0);

        }
        System.out.println("****************************");
        //print("attempt: {} hit-counts {} miss-counts: {}".format(self.current_test_attempt, hit_count, miss_count))
        System.out.println(String.format("attempt: %.0f hit-counts %.0f miss-counts: %.0f",this.current_test_attempt,hit_count,miss_count));
        System.out.println("****************************");
    }
    public void print_matrix(double [][]data){
        for(int i=0;i<data.length;i++){
            for(int j=0;j<data[i].length;j++){
                System.out.print(data[i][j]+" ");
            }
            System.out.println();

        }
    }

    public void print_arraylist(List<Double> one){
        for(double tmp:one){
            System.out.print(tmp+" ");
        }
    }

    public double LongOrDoubleToDouble(Object data){
        double result;
        if(data instanceof Double){
            result=(double) data;
        }
        else {
            result= long_to_double((Long) data);
        }
        return  result;
    }

}
