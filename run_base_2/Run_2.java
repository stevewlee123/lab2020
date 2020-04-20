package run_base2;


import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.util.*;

public class Run_2 {
    public static void sequential_recommendation(
            String data_str, String course_num, String model_str, double skill_dim, double concept_dim, double lambda_s,
            double lambda_t, double lambda_q, double lambda_bias, double lr, double penalty_weight, double markovian,
            double max_iter, String output_path,String log_file,double test_start_attempt,
            double test_end_attempt,double top_k) {

        try {
            String filename = String.format("data/%s/%s/edurec_train_test_2.json", data_str, course_num);
            JSONParser jsonParser = new JSONParser();
            JSONObject jsonObject = (JSONObject) jsonParser.parse(new FileReader(filename));
            Long num_attempts = (Long) jsonObject.get("num_attempts");
            Long num_users = (Long) jsonObject.get("num_users");
            Long num_questions = (Long) jsonObject.get("num_questions");
            Long num_discussions = (Long) jsonObject.get("num_discussions");


//            JSONArray train = (JSONArray) jsonObject.get("train");
//            JSONArray test = (JSONArray) jsonObject.get("test");
            JSONObject test_users = (JSONObject) jsonObject.get("test_users");
            JSONArray all_data = (JSONArray) jsonObject.get("all_data");
            JSONObject users_data = (JSONObject) jsonObject.get("users_data");
            JSONObject question_score_dict = (JSONObject) jsonObject.get("question_score_dict");
            JSONObject next_questions_dict = (JSONObject) jsonObject.get("next_questions_dict");
//

            System.out.println(jsonObject.keySet());
            System.out.println("==========================================");

            Config model_config = new Config(num_users, num_questions, num_discussions, num_attempts, all_data,
                    users_data, question_score_dict, next_questions_dict, test_users,
                    skill_dim, concept_dim, lambda_s, lambda_t, lambda_q, lambda_bias, lr,
                    penalty_weight, markovian, max_iter, log_file, test_start_attempt, top_k, 0.01);


            // System.out.println("==========================================");

            //  System.out.println(model_config.test_users_logged_testing_perf.keySet());
            ArrayList<List<Double>> train_set = model_config.train_data_set(all_data, test_start_attempt);
            ArrayList<List<Double>> test_set = model_config.test_data_set(all_data, test_start_attempt);

            //System.out.println(   model_config.test_users_logged_before_testing_perf.keySet());

            //System.out.println("==========================================");

            SimpleTutor tutor = new SimpleTutor(model_config);
            //    System.out.println(   tutor.test_users_logged_before_testing_perf .keySet());
           for(double att =test_start_attempt; att<test_end_attempt;att++)
           {

                tutor.current_test_attempt=att;
                tutor.lr=lr;
                tutor.training();
               for (int i=0;i<test_set.size();i++){
                   List<Double> temp=test_set.get(i);

                   if(temp.get(1)==tutor.current_test_attempt){
                       tutor.train_data.add(temp);
                   }

                   double max=temp.get(1)-tutor.markovian_steps;
                   max=Math.max(max,0);

                   for (double j=max; j<temp.get(1);j++){
                       List<Double> one=new ArrayList<Double>();
                       one.add(temp.get(0));one.add(j);one.add(temp.get(2));
                       tutor.train_data_markovian.add(one);

                   }

                   double min=temp.get(1)+tutor.markovian_steps+1;
                   min=Math.min(tutor.num_attempts,min);
                   for (double k=temp.get(1)+1; k<min;k++){
                       List<Double> one=new ArrayList<Double>();
                       one.add(temp.get(0));one.add(k);one.add(temp.get(2));
                       tutor.train_data_markovian.add(one);

                   }

               }
               tutor.generate_next_items();

           }


           JSONArray a = new JSONArray ();
            JSONArray  b = new JSONArray ();
            JSONArray  c = new JSONArray ();
            JSONArray  d = new JSONArray ();


            Iterator key_iter = tutor.test_users_perf.keySet().iterator();
            List<Double> test_users_perf_keys = new ArrayList<Double>();
            while (key_iter.hasNext()) {
                test_users_perf_keys.add((double) key_iter.next());
            }
            Collections.sort(test_users_perf_keys);


            for (double user : test_users_perf_keys) {
                List<Double> test_users_perf_user = (List<Double>) tutor.test_users_perf.get(user);
                double mean_a = get_mean(test_users_perf_user);
                a.add(mean_a);

                //  System.out.println(tutor.test_users_logged_testing_perf.keySet());
                List<Double> test_users_logged_testing_perf_user = (List<Double>) tutor.test_users_logged_testing_perf.get(user);
                double mean_b = get_mean(test_users_logged_testing_perf_user);
                b.add(mean_b);

                double normalized_discounted_perf = 0;
                double perf = 0;


                JSONObject test_users_historical_records_user = (JSONObject) tutor.test_users_historical_records.get(user);
                System.out.println("historical records :" + test_users_historical_records_user);

                Iterator test_users_historical_records_user_key = test_users_historical_records_user.keySet().iterator();

                while (test_users_historical_records_user_key.hasNext())
                {
                    double question = (double) test_users_historical_records_user_key.next();
                    JSONArray u_q = (JSONArray) test_users_historical_records_user.get(question);
                    double final_question_score = (double) u_q.get(u_q.size() - 1);
                    perf += final_question_score;

                    int num_trials = u_q.size();

                    normalized_discounted_perf += final_question_score / (Math.log(num_trials + 1) / Math.log(2));
                    System.out.println(String.format("%f,%f,%d,%f",question,final_question_score,num_trials,normalized_discounted_perf));

                }
                //System.exit(0);
                c.add(normalized_discounted_perf);
                d.add(perf);

                double nbch = mean_a;

                List<Double> question_list = new ArrayList<Double>();

                JSONArray users_data_user = (JSONArray) tutor.users_data.get(Integer.toString((int) user));

                for (int i = 0; i < users_data_user.size(); i++) {
                    JSONArray one = (JSONArray) users_data_user.get(i);
                    double question = long_to_double((long)one.get(2));
                    if (!question_list.contains(question)) {
                        question_list.add(question);
                    }
                }


                System.out.println(String.format("student %.0f, number of attempts: %d, number of questions: %d", user,
                        users_data_user.size(), question_list.size()   ));

                //print("propensity ndch {:.5f} and student real performance: {:.5f}".format(
                      //  np.mean(tutor.test_users_perf[user]), np.mean(tutor.test_users_logged_testing_perf[user])))

                System.out.println(String.format("propensity ndch %.5f and student real performance: %.5f",nbch,mean_b));

                System.out.println("------------------------------");

            }

            JSONObject result=new JSONObject();
            result.put("propensity_ndch",a);
            result.put("logged_avg_perf",b);
            result.put("logged_discount_perf",c);
            result.put("logged_last_perf",d);
            try (FileWriter file = new FileWriter("employees.json")) {

                file.write(result.toJSONString());
                file.flush();

            } catch (IOException e) {
                e.printStackTrace();
            }



        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }


    }
    public static void run_morf(){
        String data_str="morf";
        String course_num="Quiz_Only_Discussion";
        String model_str="simple";


        double skill_dim=45;
        double concept_dim = 5;
        double lambda_s = 0.1;
        double  lambda_t = 0.001;
        double lambda_q = 0.;
        double lambda_bias = 0;
        double lr = 0.1;
        double penalty_weight = 0.4;
        double markovian = 1;
        double max_iter = 30;
        double test_start_attempt=2;
        double test_end_attempt=25;
        double top_k=3;
        String log_file=null;
        String output_path=String.format("results/%s/%s_%s_final_test_run2.csv", data_str,course_num, model_str);

        sequential_recommendation(data_str, course_num, model_str, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                lambda_bias, lr, penalty_weight, markovian, max_iter, output_path,log_file,test_start_attempt,
                test_end_attempt, top_k);
    }
    public static void print_arraylist(List<Double> one){
        for(double tmp:one){
            System.out.print(tmp+" ");
        }
    }

    public static double get_mean(List<Double> data){
        double mean=0;
        for (double tmp:data){
            mean+=tmp;
        }
        return mean/data.size();
    }
    public static double long_to_double(Long data){
        double result;
        Long tmp_num = data;
        result = tmp_num.doubleValue();
        return result;
    }


    public static void main( String[] args) {
        run_morf();

    }
}
