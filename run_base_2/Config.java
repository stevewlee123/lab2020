package run_base2;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class Config<T> {
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
    public double top_k;
    public double start_test_attempt;
    public JSONArray all_data;
    public JSONObject test_users;
    public JSONObject users_data;
    public JSONObject test_users_logged_perf;
    public JSONObject next_questions_dict;
    public JSONObject question_score_dict;
    public JSONObject test_users_records;
    public JSONObject test_users_logged_testing_perf;

    public JSONObject test_users_logged_before_testing_perf;


    public Config(T num_users, T num_questions, T num_discussions, T num_attempts, JSONArray all_data,
                  JSONObject users_data , JSONObject question_score_dict,
                  JSONObject next_questions_dict,JSONObject test_users, T skill_dim,T concept_dim, T lambda_s, T lambda_t,
                  T lambda_q, T lambda_bias, T lr, T penalty_weight, T markovian_steps,  T max_iter,
                   T log_file, double test_start_attempt, double top_k,T tol) {
        this.num_users = num_users;
        this.num_questions = num_questions;
       // this.num_discussions = num_discussions;
        this.num_attempts = num_attempts;
        this.num_skills = skill_dim;
        this.num_concepts = concept_dim;
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
        this.test_users=test_users;
        this.users_data=users_data;

        this.all_data=all_data;
        this.top_k=top_k;
        this.start_test_attempt=test_start_attempt;
       // this.test_users_logged_perf=test_users_logged_perf;
        this.next_questions_dict=next_questions_dict;
        this.question_score_dict=question_score_dict;

    }

    public double long_to_double(Long data){
        double result;
        Long tmp_num = data;
        result = tmp_num.doubleValue();
        return result;
    }

    public ArrayList<List<Double>> train_data_set(JSONArray all_data,double test_start_attempt)
    {
        ArrayList<List<Double>> train_set= new ArrayList<List<Double>>();
        for(int i=0;i<all_data.size();i++)
        {
            JSONArray temp = (JSONArray) all_data.get(i);
            double student=long_to_double((Long)temp.get(0));
            double attempt=long_to_double((Long)temp.get(1));
            if(test_users.get(Integer.toString((int)student))==null || long_to_double((Long)temp.get(1))<test_start_attempt){


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
        }
        //this.train=new ArrayList<List<Double>>();
        this.train=(ArrayList<List<Double>>) train_set.clone();
        return train_set;

    }

    public ArrayList<List<Double>> test_data_set(JSONArray all_data, double test_start_attempt){
        JSONObject test_users_records=new JSONObject();
        JSONObject test_users_logged_testing_perf=new JSONObject();
        JSONObject test_users_logged_before_testing_perf=new JSONObject();
        ArrayList<List<Double>> test_set= new ArrayList<List<Double>>();
        for(int i=0;i<all_data.size();i++)
        {

            JSONArray temp = (JSONArray) all_data.get(i);
            double student=long_to_double((Long)temp.get(0));
            double attempt=long_to_double((Long)temp.get(1));
            if(this.test_users.get(Integer.toString((int)student))!=null)
            {

                JSONObject temp1;
                if(test_users_records.get(student)==null){
                    temp1=new JSONObject();

                    test_users_records.put(student,temp1);
                }
                else{
                    temp1=( JSONObject)test_users_records.get(student);
                }

                //System.out.println(test_users_records.keySet());
                temp1.put(attempt,long_to_double((Long)temp.get(2)));
            }
            if(this.test_users.get(Integer.toString((int)student))!=null  && attempt<test_start_attempt)
            {
                JSONArray temp1;
                if(test_users_logged_before_testing_perf.get(long_to_double((Long)temp.get(0)))==null){
                    temp1=new JSONArray();
                    test_users_logged_before_testing_perf.put(student,temp1);
                }
                else{
                    temp1=( JSONArray)test_users_logged_before_testing_perf.get(student);
                }

                temp1.add((double)temp.get(3));
            }
            if(this.test_users.get(Integer.toString((int)student))!=null  && attempt >= test_start_attempt)
            {

                ArrayList<Double> each_one = new ArrayList<Double>();
                for (int j = 0; j < 4; j++)
                {
                    double a;
                    if(temp.get(j) instanceof Double){
                        a =(double) temp.get(j);
                    }
                    else {
                        a= long_to_double((Long) temp.get(j));
                    }
                    each_one.add(a);
                }
                 test_set.add(each_one);


               List<Double> temp1;
                if(test_users_logged_testing_perf.get(long_to_double((Long)temp.get(0)))==null)
                {
                   // System.out.println(1);
                    temp1=new ArrayList<Double>();
                    test_users_logged_testing_perf.put(student,temp1);
                }
                else{
                    temp1=( List<Double>)test_users_logged_testing_perf.get(student);
                }

                temp1.add((double)temp.get(3));
            }


        }


        //this.test=new ArrayList<List<Double>>();
        this.test=(ArrayList<List<Double>>)test_set.clone();
        this.test_users_records=(JSONObject) test_users_records.clone();
        this.test_users_logged_testing_perf=(JSONObject) test_users_logged_testing_perf.clone();

        this.test_users_logged_before_testing_perf=(JSONObject) test_users_logged_before_testing_perf.clone();

        return test_set;
    }

}
