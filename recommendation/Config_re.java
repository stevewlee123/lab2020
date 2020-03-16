package lab2020_recom;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class Config_re<T> {
        public T num_users;
        public T num_questions;
        public T num_discussions;
        public T num_attempts;
        public T num_skills;
        public T num_concepts;
        public T lambda_s;
        public T lambda_t;
        public T lambda_q;
        public T lambda_bias;
        public T penalty_weight;
        public T markovian_steps;
        public T trade_off;
        public T lr;
        public T tol;
        public T max_iter;
        public T log_file;
        public ArrayList<List<Double>> train;
        public ArrayList<List<Double>> test;
        public JSONObject test_users;
        public JSONObject users_data;
        public JSONObject test_users_logged_perf;
        public JSONObject next_questions_dict;
        public JSONObject question_score_dict;

        public Config_re(T num_users, T num_questions, T num_discussions, T num_attempts,
                         JSONObject test_users_logged_perf,JSONObject users_data ,JSONObject question_score_dict,
                         JSONObject next_questions_dict, T num_skills, T num_concepts, T lambda_s, T lambda_t,
                         T lambda_q, T lambda_bias,T lr, T penalty_weight,T markovian_steps, T tol, T max_iter,
                         T data_reousrce,T trade_off, T inner_fold,T log_file) {
            this.num_users = num_users;
            this.num_questions = num_questions;
            this.num_discussions = num_discussions;
            this.num_attempts = num_attempts;
            this.num_skills = num_skills;
            this.num_concepts = num_concepts;
            this.lambda_s = lambda_s;
            this.lambda_t = lambda_t;
            this.lambda_q = lambda_q;
            this.lambda_bias = lambda_bias;
            this.penalty_weight=penalty_weight;
            this.markovian_steps=markovian_steps;
            this.trade_off = trade_off;
            this.lr = lr;
            this.tol=tol;
            this.max_iter=max_iter;
            this.log_file = log_file;
            this.users_data=users_data;
            this.test_users_logged_perf=test_users_logged_perf;
            this.next_questions_dict=next_questions_dict;
            this.question_score_dict=question_score_dict;

        }

        public double long_to_double(Long data){
            double result;
            Long tmp_num = data;
            result = tmp_num.doubleValue();
            return result;
        }

        public ArrayList<List<Double>> train_data_set(JSONArray train)
        {
            ArrayList<List<Double>> train_set= new ArrayList<List<Double>>();
            for(int i=0;i<train.size();i++)
            {
                JSONArray temp = (JSONArray) train.get(i);
                ArrayList<Double> each_one = new ArrayList<Double>();
                for (int j = 0; j < 4; j++)
                {
                    double a;
                    if(temp.get(j) instanceof Double){
                        a =(double) temp.get(j);
                    }
                    else {
                       a=long_to_double ( (Long)temp.get(j) );
                    }

                    each_one.add(a);
                }
                train_set.add(each_one);

            }
            this.train=train_set;
            return train_set;

        }

        public ArrayList<List<Double>> test_data_set(JSONArray test){
            JSONObject test_users=new JSONObject();
            ArrayList<List<Double>> test_set= new ArrayList<List<Double>>();
            for(int i=0;i<test.size();i++)
            {
                JSONArray temp1 = (JSONArray) test.get(i);
                ArrayList<Double> each_one = new ArrayList<Double>();
                for (int j = 0; j < 4; j++)
                {
                    double a;
                    if(temp1.get(j) instanceof Double){
                        a =(double) temp1.get(j);
                    }
                    else {
                        a= long_to_double((Long) temp1.get(j));
                    }
                    each_one.add(a);
                }
                test_set.add(each_one);

                if(test_users.get(long_to_double((Long)temp1.get(0)))==null)
                {
                    JSONObject temp2=new JSONObject();
                    test_users.put(long_to_double((Long)temp1.get(0)),temp2);
                }
                JSONObject temp3=( JSONObject)test_users.get(long_to_double((Long)temp1.get(0)));
                temp3.put(long_to_double((Long)temp1.get(1)),long_to_double((Long)temp1.get(2)));
            }
            this.test=test_set;
            this.test_users=test_users;
            return test_set;
        }

    }

