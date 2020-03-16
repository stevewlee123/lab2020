package lab2020_recom;

import lab2020.Config;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Run_base {

        public static void sequential_recommendation(
                String data_str, String course_num, String model_str, double skill_dim, double concept_dim, double lambda_s,
                double lambda_t, double lambda_q, double lambda_bias, double lr, double penalty_weight, double markovian,
                double max_iter, String output_path,String log_file){
            JSONObject perf_dict=new JSONObject();
            try {
                String filename = String.format("data/%s/%s/edurec_train_test.json", data_str, course_num);
                JSONParser jsonParser = new JSONParser();
                JSONObject jsonObject = (JSONObject) jsonParser.parse(new FileReader(filename));
                Long num_attempts = (Long) jsonObject.get("num_attempts");
                Long num_users = (Long) jsonObject.get("num_users");
                Long num_questions = (Long) jsonObject.get("num_questions");
                Long num_discussions= (Long) jsonObject.get("num_discussions");
                JSONArray train = (JSONArray) jsonObject.get("train");
                JSONArray test = (JSONArray) jsonObject.get("test");
                JSONObject test_users =  (JSONObject)jsonObject.get("test_users");
                JSONObject test_users_logged_perf =  (JSONObject)jsonObject.get("test_users_logged_perf");
                JSONObject users_data = (JSONObject) jsonObject.get("users_data");
                JSONObject question_score_dict= (JSONObject) jsonObject.get("question_score_dict");
                JSONObject next_questions_dict= (JSONObject) jsonObject.get("next_questions_dict");

                System.out.println("==========================================");
                System.out.println(jsonObject.keySet());
                System.out.println("==========================================");
                Config_re model_config = new Config_re(num_users, num_questions, num_discussions, num_attempts,
                        test_users_logged_perf, users_data ,question_score_dict,next_questions_dict,
                        skill_dim,concept_dim, lambda_s, lambda_t, lambda_q,lambda_bias,lr,
                        penalty_weight, markovian,1e-2,max_iter,null,1.0,null,null);

                ArrayList<List<Double>> train_set=model_config.train_data_set(train);
                ArrayList<List<Double>> test_set=model_config.test_data_set(test);

                SimpleTutor tutor=new SimpleTutor(model_config);
                tutor.set_train_data_markovian();
                tutor.set_current_questions_states_scores_perf();


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



                System.out.println("test start attempt: "+test_start_attempt);
                int total_test_count=0;
                double sum_square_error=0;double sum_abs_error=0;double sum_percent_error=0;double sum_abs_percent_error=0;

                for(int att=test_start_attempt;att<tutor.num_attempts;att++){
                    tutor.current_test_attempt=(double)att;

                    tutor.training();


                    for(int j=0; j< test_set.size() ; j++)
                    {
                    List<Double> one=test_set.get(j);
                        if(one.get(1)==tutor.current_test_attempt)
                        {
                            tutor.train_data.add(one);// student attempt question obs
                        }
                    }
                    tutor.generate_next_items();

                }

//                Object[] tmp_five=tutor.test_users_perf.keySet().toArray();
//                for( Object one :tmp_five){
//                    JSONArray two=(JSONArray)tutor.users_data.get(String.format("%d",(int)(double)one));
//                    System.out.print(one+" , ");
//                    System.out.print(array_list_mean((List<Double>)tutor.test_users_perf.get( (double)one )));
//                    System.out.println(" , "+array_list_mean((List<Double>)tutor.test_users_logged_perf.
//                            get(String.format("%d",(int)(double)one))));
//                    System.out.println(tutor.users_data.get(String.format("%d",(int)(double)one)));
//                    System.out.println("number of attempts: "+two.size());
//                    //System.out.print(", "+array_list_mean((List<Double>) tutor.test_users_logged_perf.get(String.format("%f",one))));
//
//                    List<Long>question_list = new ArrayList<Long>();
//
//                    for(int m=0;m<two.size();m++){
//                        JSONArray each_one=( JSONArray)two.get(m);
//                        question_list.add((long)each_one.get(2));
//                    }
//                    Set<Long> set = new HashSet<>(question_list);
//                    System.out.println("number of questions: "+set.size());
 //               }
            }catch (FileNotFoundException e) {
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
            double skill_dim=10;
            double concept_dim = 3;
            double lambda_s = 0.;
            double  lambda_t = 0.001;
            double lambda_q = 0.;
            double lambda_bias = 0;
            double lr = 0.1;
            double penalty_weight = 1;
            double markovian = 1;
            double max_iter = 30;
            String log_file=null;
            String output_path=String.format("results/%s/%s_%s_final_test.csv", data_str,course_num, model_str);
            sequential_recommendation(data_str, course_num, model_str, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                    lambda_bias, lr, penalty_weight, markovian, max_iter, output_path,log_file);
        }

        public static void main( String[] args) {
            run_morf();

        }
    public static void print_matrix(double [][]data){
        for(int i=0;i<data.length;i++){
            for(int j=0;j<data[i].length;j++){
                System.out.print(data[i][j]+" ");
            }
            System.out.println();

        }
    }
    public static void print_array(double []data){
            for(double one :data){
                System.out.print(one+" ");
            }
    }
    public static double array_list_mean(List<Double> data){
            double mean=0;
            for(double one: data){
                mean+=one;
            }
            mean/=data.size();
            return mean;
    }

}


