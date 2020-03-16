package lab2020;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Rank_cross_validation implements Runnable{
    private Lock lock = new ReentrantLock(true);
    private double[] para_double;
    private String[] para_string;
    private int task_num;
    public Rank_cross_validation(double[] para_double,String[] para_string,int t_s){
        this.para_double=para_double;
        this.para_string=para_string;
        this.task_num=t_s;
    }
    public void run()
    {
        try {
           // System.out.println(String.format("Task_%d is begin",this.task_num));
            cross_validation();
           // System.out.println(String.format("Task_%d is end",this.task_num));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void check_progress(String data_str, String course_num, String model_str,double markovian){
        String filename = String.format("results/%s/%s_%s_markovian_%f_cross_val.csv", data_str, course_num, markovian);
    }
    public void cross_validation() throws IOException {
        String data_str=this.para_string[0]; String course_num=this.para_string[1];
        String model_str=this.para_string[2]; String output_path=this.para_string[3];
        double skill_dim=this.para_double[0]; double concept_dim=this.para_double[1]; double lambda_s=this.para_double[2];
        double lambda_t=this.para_double[3]; double lambda_q=this.para_double[4]; double lambda_bias=this.para_double[5];
        double lr=this.para_double[6];double penalty_weight=this.para_double[7]; double markovian=this.para_double[8];
        double max_iter=this.para_double[9]; double resource=this.para_double[10];


        JSONObject perf_dict=new JSONObject();
        for (int fold = 1; fold < 2; fold++) {
            JSONObject temp=new JSONObject();
            if(perf_dict.get(fold)==null)
            {
                perf_dict.put(fold,temp);
            }
            try {
                String filename = String.format("data/%s/%s/%d_train_test.json", data_str, course_num, fold);
                JSONParser jsonParser = new JSONParser();
                JSONObject jsonObject = (JSONObject) jsonParser.parse(new FileReader(filename));
                Long num_users = (Long) jsonObject.get("num_users");
                Long num_questions = (Long) jsonObject.get("num_quizs");
                Long num_discussions = (Long) jsonObject.get("num_disicussions");
                Long num_attempts = (Long) jsonObject.get("num_attempts");
                JSONArray train = (JSONArray) jsonObject.get("train");
                JSONArray test = (JSONArray) jsonObject.get("test");
                JSONArray cross_train = (JSONArray) jsonObject.get("cross_train");
                JSONArray cross_validation = (JSONArray) jsonObject.get("cross_validation");
//        Object tem=train.get(0);
//        System.out.println(tem.get(0));
                //System.out.println(fold);
                Config model_config = new Config(num_users, num_questions, num_discussions, num_attempts, skill_dim, concept_dim,
                        lambda_s, lambda_t, lambda_q, null, 1e-3, lambda_bias, penalty_weight, markovian, 1.0,
                        lr, max_iter, null);
                boolean inter_fold = false;
                ArrayList<List<Double>>  train_set;
                ArrayList<List<Double>>  test_set;
                if (!inter_fold) {
                    train_set = model_config.data_set(train);
                    test_set = model_config.data_set(test);
                    model_config.train = train_set;
                } else {
                    train_set = model_config.data_set(cross_train);
                    test_set = model_config.data_set(cross_validation);
                    model_config.train = train_set;
                }
                //创建问题
                RankObsModel model = new RankObsModel(model_config);

                //printArr(test_set);
                ArrayList<List<Double>>  tmp=(ArrayList<List<Double>>)test_set.clone();
                Collections.sort(tmp, new Comparator<List<Double>>()
                {
                    public int compare(List<Double> d1, List<Double> d2)
                    {
                        Double a=d1.get(1);
                        Double b=d2.get(1);
                        return  a.compareTo(b);
                    }
                });
                //System.out.println();
                // printArr(test_set);
                //System.exit(0);
                int test_start_attempt=(tmp.get(0).get(1)).intValue();

                int total_test_count=0;
                double sum_square_error=0;double sum_abs_error=0;double sum_percent_error=0;double sum_abs_percent_error=0;
                //System.out.println(1);
                //printArr(tmp);
                for(int i=test_start_attempt;i<model.num_attempts;i++){
                    //System.out.println(String.format("\n\nfold: %d, test attempt: %d/%d",fold,i,model.num_attempts-1));
                    model.current_test_attempt=i;
                    //model.lr=lr;
                    model.training();
                    //System.exit(0);
                    List<List<Double>>val_data=new ArrayList<List<Double>>();

                    //System.out.println(String.format( "train size: %d",model.train_data.size()));
                    //  System.out.println(model.bias_s.length);
                    for(int j=0;j<test_set.size();j++){
                        List<Double> one=new ArrayList<>(test_set.get(j).subList(0,4));
                        if(one.get(1).intValue()==(int)model.current_test_attempt){
                            val_data.add(one);
                            model.train_data.add(one);
                           // model.all_obs.add(one.get(3));
//                            double obs=one.get(3);
//                            if(obs==0.0){
//                                obs=1e-6;
//                            }
//                            if(model_str.equals("rank_obs")){
//                                model.obs_tensor[one.get(0).intValue()][one.get(1).intValue()][one.get(2).intValue()]=obs;
//                                model.train_order.add(new ArrayList<>(one.subList(0,3)));
//                            }

                        }
                    }
                    //System.out.println(String.format( "test size: %d",val_data.size()));
                    //  System.out.println(model.bias_s.length);
                    double []val_perf=model.testing(val_data);
//                    model.global_avg=0;
//                    for(int j=0;j<model.all_obs.size();j++){
//                        model.global_avg+=(double)model.all_obs.get(j);
//                    }
//                    model.global_avg/=model.all_obs.size();
//                    System.out.println(String.format("global_avg: %f",model.global_avg));
//
//                    model.S=new double[model.num_users][model.num_skills];
//                    model.T=new double[model.num_skills][model.num_attempts][model.num_concepts];
//                    model.Q=new double[model.num_concepts][model.num_questions];
//
//                    model.bias_s=new double[model.num_users];
//                    model.bias_q=new double[model.num_questions] ;
//                    model.bias_a=new double[model.num_attempts];
                    if(temp.get(i)==null){
                        JSONArray test_perf_tmp=new JSONArray();
                        test_perf_tmp.add(val_perf[0]);
                        test_perf_tmp.add(val_perf[1]);
                        test_perf_tmp.add(val_perf[2]);
                        test_perf_tmp.add(val_perf[3]);
                        test_perf_tmp.add(val_perf[4]);
                        temp.put(i,test_perf_tmp);
                    }
                    double val_count=val_perf[0];double _rmse=val_perf[1];double _mae=val_perf[2];double _mpe=val_perf[3];
                    double _mape=val_perf[4];

                    sum_square_error += Math.pow(_rmse,2) * val_count;
                    sum_abs_error += _mae * val_count;
                    sum_percent_error += _mpe * val_count;
                    sum_abs_percent_error += _mape * val_count;
                    total_test_count += val_count;


                }
                double rmse=Math.sqrt(sum_square_error/total_test_count);
                double mae=sum_abs_error/total_test_count;
                double mpe=sum_percent_error/total_test_count;
                double mape=sum_abs_percent_error/total_test_count;
                JSONArray overall_tmp=new JSONArray();
                overall_tmp.add(total_test_count);
                overall_tmp.add(rmse);
                overall_tmp.add(mae);
                overall_tmp.add(mpe);
                overall_tmp.add(mape);
                temp.put("overall",overall_tmp);
//                ObjectOutputStream oout=new ObjectOutputStream(new FileOutputStream(String.format(
//                        "outputs/%s/%s_%s_fold_%d_model.pkl",data_str, course_num, model_str, fold)));
//                oout.writeObject(model);   //写入文件
//                oout.close();


            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (ParseException e) {
                e.printStackTrace();
            }

        }
        JSONObject result=new JSONObject();
        result.put("skill_dim", skill_dim);
        result.put("concept_dim", concept_dim);
        result.put("lambda_s", lambda_s);
        result.put("lambda_t", lambda_t);
        result.put("lambda_q", lambda_q);
        result.put("lambda_bias", lambda_bias);
        result.put("lr", lr);
        result.put("penalty_weight", penalty_weight);
        result.put("markovian_steps", markovian);
        result.put("max_iter", max_iter);
        result.put("perf", perf_dict);
        lock.lock();
        FileWriter file=new FileWriter(output_path,true);
        file.write(result.toJSONString());
        file.write("\n");
        file.flush();
        lock.unlock();

    }

    public static void morf_hyperparameter_tuning(String data_str,String course_num,String model_str
    ,double markovian,double concept_dim,double num_proc) throws IOException {
        double resource=0;
        if(course_num.equals("Quiz_Only_Discussion"))
        {
            List<double[]> para_list_double=new ArrayList<double[]>();
            List<String[]> para_list_string=new ArrayList<String[]>();
            double lambda_q =0;
            double lr = 0.1;
            double max_iter = 30;
            String output_path=String.format("results/%s/%s_%s_markovian_%f_cross_val.csv", data_str,course_num, model_str, markovian);

//            try (FileWriter file = new FileWriter(output_path)) {
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
            double[]skill_array=new double[]{1, 3, 5, 7, 9};
            double[]lambda_s_k=new double[]{0,1,5,10};
            double[]lambda_t_k=new double[]{0, 1, 5};
            double[]lambda_bias_k=new double[]{0};
            double[]penalty_weight_array=new double[]{5,10};
            for(int ski=0;ski<skill_array.length;ski++ )
            {
                double skill_dim=skill_array[ski];
                concept_dim=skill_dim;
                for (int la_s_k=0;la_s_k<lambda_s_k.length;la_s_k++ )
                {
                    double lambda_s=0.001*lambda_s_k[la_s_k];
                    if(lambda_s!=0){
                        lr=0.01 ;
                    }
                    for(int la_t_k=0;la_t_k<lambda_t_k.length;la_t_k++)
                    {
                        double lambda_t=0.001*lambda_t_k[la_t_k];
                        for(int la_b_k=0;la_b_k<lambda_bias_k.length;la_b_k++)
                        {
                            double lambda_bias=0.0001*lambda_bias_k[la_b_k];
                            for(int pen_we=0;pen_we<penalty_weight_array.length;pen_we++)
                            {
                                double penalty_weight=penalty_weight_array[pen_we]*0.01;
                                String []para_string=new String[]{data_str, course_num, model_str,output_path};
                                double []para_double=new double[]{skill_dim, concept_dim,
                                        lambda_s, lambda_t, lambda_q, lambda_bias, lr,
                                        penalty_weight, markovian, max_iter,resource};
                                para_list_double.add(para_double);
                                para_list_string.add(para_string);
                            }
                        }


                    }
                }
            }
            ExecutorService pool = Executors.newFixedThreadPool((int)num_proc);

            for(int i=0;i<para_list_double.size();i++){
                String[] para_string_tmp=para_list_string.get(i);
                double[] para_double_tmp=para_list_double.get(i);
                Runnable r1 = new Rank_cross_validation(para_double_tmp,para_string_tmp,i);
                pool.execute(r1);


            }
            pool.shutdown();
        }
        else if(course_num.equals("Quiz_Only_Lecture"))
        {
            List<double[]> para_list_double=new ArrayList<double[]>();
            List<String[]> para_list_string=new ArrayList<String[]>();
           double lambda_q =0;double lambda_e = 0;
            double lr = 0.1;
            double max_iter = 30;
            String output_path=String.format("results/%s/%s_%s_concept_%f_cross_val.csv", data_str,course_num, model_str, concept_dim);

            try (FileWriter file = new FileWriter(output_path)) {
            } catch (IOException e) {
                e.printStackTrace();
            }
            double[]skill_array=new double[]{25, 30, 35, 40};
            double[]lambda_s_k=new double[]{0};
            double[]lambda_t_k=new double[]{0, 1, 5};
            double[]lambda_bias_k=new double[]{0};
            double[]penalty_weight_array=new double[]{0, 0.05, 0.1, 0.5, 1, 2};
            for(int ski=0;ski<skill_array.length;ski++ )
            {
                double skill_dim=skill_array[ski];
                for (int la_s_k=0;la_s_k<lambda_s_k.length;la_s_k++ )
                {
                    double lambda_s=0.001*lambda_s_k[la_s_k];
                    if(lambda_s!=0){
                        lr=0.01 ;
                    }
                    for(int la_t_k=0;la_t_k<lambda_t_k.length;la_t_k++)
                    {
                        double lambda_t=0.001*lambda_t_k[la_t_k];
                        for(int la_b_k=0;la_b_k<lambda_bias_k.length;la_b_k++)
                        {
                            double lambda_bias=0.0001*lambda_bias_k[la_b_k];
                            for(int pen_we=0;pen_we<penalty_weight_array.length;pen_we++)
                            {
                                double penalty_weight=penalty_weight_array[pen_we];
                                String []para_string=new String[]{data_str, course_num, model_str,output_path};
                                double []para_double=new double[]{skill_dim, concept_dim,
                                        lambda_s, lambda_t, lambda_q, lambda_bias, lr,
                                        penalty_weight, markovian, max_iter,resource};
                                para_list_double.add(para_double);
                                para_list_string.add(para_string);
                            }
                        }


                    }
                }
            }
            ExecutorService pool = Executors.newFixedThreadPool((int)num_proc);

            for(int i=0;i<para_list_double.size();i++){
                String[] para_string_tmp=para_list_string.get(i);
                double[] para_double_tmp=para_list_double.get(i);
                Runnable r1 = new Rank_cross_validation(para_double_tmp,para_string_tmp,i);
                pool.execute(r1);


            }
            pool.shutdown();
        }

    }

    public static void morf() throws IOException {
        String data_str="morf";
        String course_num="Quiz_Only_Lecture";
        String model_str="rank_obs";
       double num_proc = 25;
       double markovian = 1;
       double concept_dim = 8;
        morf_hyperparameter_tuning(data_str, course_num, model_str, markovian, concept_dim, num_proc);


      //  String output_path=String.format("%s_%s_markovian_%d_cross_val.csv", data_str,course_num, model_str,markovian);
//        String log_file=null;
//        single_rank_exp(data_str,course_num,model_str,output_path,skill_dim,concept_dim,lambda_s,
//                lambda_t,lambda_q,lambda_bias,lr,penalty_weight,markovian,max_iter,resource);

    }

    public static void main(String[] args) throws IOException {
        morf();

    }
}
