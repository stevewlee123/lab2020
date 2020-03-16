package lab2020;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.util.*;

public class Run_rank  implements Serializable{


    public static void single_rank_exp(String data_str, String course_num, String model_str, String output_path,
                                       double skill_dim, double concept_dim, double lambda_s,
                                       double lambda_t, double lambda_q, double lambda_bias, double lr,
                                       double penalty_weight, double markovian, double max_iter, double resource) {
        JSONObject perf_dict=new JSONObject();
        for (int fold = 1; fold < 6; fold++) {
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
                int test_start_attempt=(tmp.get(0).get(1)).intValue();

                int total_test_count=0;
                double sum_square_error=0;double sum_abs_error=0;double sum_percent_error=0;double sum_abs_percent_error=0;
                //System.out.println(1);
                //printArr(tmp);
                for(int i=test_start_attempt;i<model.num_attempts;i++){
                    System.out.println(String.format("\n\nfold: %d, test attempt: %d/%d",fold,i,model.num_attempts-1));
                    model.current_test_attempt=i;
                    model.lr=lr;
                    model.training();
                    //System.exit(0);
                    List<List<Double>>test_data=new ArrayList<List<Double>>();

                    System.out.println(String.format( "train size: %d",model.train_data.size()));
                  //  System.out.println(model.bias_s.length)
                    for(int j=0;j<test_set.size();j++){
                        List<Double> one=new ArrayList<>(test_set.get(j).subList(0,4));
                        if(one.get(1).intValue()==(int)model.current_test_attempt){
                            test_data.add(one);
                            model.train_data.add(one);
                            model.all_obs.add(one.get(3));
                            double obs=one.get(3);
                            if(obs==0.0){
                                obs=1e-6;
                            }
                            if(model_str.equals("rank_obs")){
                                model.obs_tensor[one.get(0).intValue()][one.get(1).intValue()][one.get(2).intValue()]=obs;
                                model.train_order.add(new ArrayList<>(one.subList(0,3)));
                            }

                        }
                    }
                    System.out.println(String.format( "test size: %d",test_data.size()));
                  //  System.out.println(model.bias_s.length);
                    double []test_perf=model.testing(test_data);
                    model.global_avg=0;
                    for(int j=0;j<model.all_obs.size();j++){
                        model.global_avg+=(double)model.all_obs.get(j);
                    }
                    model.global_avg/=model.all_obs.size();
                    System.out.println(String.format("global_avg: %f",model.global_avg));

                    model.S=new double[model.num_users][model.num_skills];
                    model.T=new double[model.num_skills][model.num_attempts][model.num_concepts];
                    model.Q=new double[model.num_concepts][model.num_questions];

                    model.bias_s=new double[model.num_users];
                    model.bias_q=new double[model.num_questions] ;
                    model.bias_a=new double[model.num_attempts];
                    if(temp.get(i)==null){
                        JSONArray test_perf_tmp=new JSONArray();
                        test_perf_tmp.add(test_perf[0]);
                        test_perf_tmp.add(test_perf[1]);
                        test_perf_tmp.add(test_perf[2]);
                        test_perf_tmp.add(test_perf[3]);
                        test_perf_tmp.add(test_perf[4]);
                        temp.put(i,test_perf_tmp);
                    }
                    double test_count=test_perf[0];double _rmse=test_perf[1];double _mae=test_perf[2];double _mpe=test_perf[3];
                    double _mape=test_perf[4];

                    sum_square_error += Math.pow(_rmse,2) * test_count;
                    sum_abs_error += _mae * test_count;
                    sum_percent_error += _mpe * test_count;
                    sum_abs_percent_error += _mape * test_count;
                    total_test_count += test_count;


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
                ObjectOutputStream oout=new ObjectOutputStream(new FileOutputStream(String.format(
                        "outputs/%s/%s_%s_fold_%d_model.pkl",data_str, course_num, model_str, fold)));
                oout.writeObject(model);   //写入文件
                oout.close();


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
        try (FileWriter file = new FileWriter(output_path)) {

            file.write(result.toJSONString());
            file.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void printArr(ArrayList<List<Double>>  arr) {
        for (int x = 0; x < arr.size(); x++) {
            System.out.print(arr.get(x));
//            for (int y = 0; y < arr.get(x).length; y++) {
//                System.out.print(arr.get(x)[y] + " ");
//            }
            System.out.print(",");
        }
    }



    public static void run_morf(){
        String data_str="morf";
        String course_num="Quiz_Only_Discussion";
        String model_str="rank_obs";
        double skill_dim=35;
        double concept_dim = 5;
        double lambda_s = 0.;
        double  lambda_t = 0.001;
        double lambda_q = 0.;
        double lambda_bias = 0;
        double lr = 0.1;
        double penalty_weight = 2;
        double markovian = 1;
        double max_iter = 30;
        double resource = 0;

        String output_path=String.format("results/%s/%s_%s_final_test.csv", data_str,course_num, model_str);
        String log_file=null;
        single_rank_exp(data_str,course_num,model_str,output_path,skill_dim,concept_dim,lambda_s,
                lambda_t,lambda_q,lambda_bias,lr,penalty_weight,markovian,max_iter,resource);

    }
    public static void main( String[] args) {
    run_morf();

    }

}
