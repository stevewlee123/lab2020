package lab2020;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import java.io.*;
import java.util.*;
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
    public T lambda_e;
    public T lambda_bias;
    public T penalty_weight;
    public T markovian_steps;
    public T trade_off;
    public T lr;
    public T tol;
    public T max_iter;
    public T log_file;
    public T train;

    public Config(T num_users, T num_questions, T num_discussions, T num_attempts, T num_skills, T num_concepts,
                  T lambda_s, T lambda_t, T lambda_q, T lambda_e,T tol, T lambda_bias, T penalty_weight,T markovian_steps,
                  T trade_off, T lr, T max_iter,T log_file) {

        this.num_users = num_users;
        this.num_questions = num_questions;
        this.num_discussions = num_discussions;
        this.num_attempts = num_attempts;
        this.num_skills = num_skills;
        this.num_concepts = num_concepts;
        this.lambda_s = lambda_s;
        this.lambda_t = lambda_t;
        this.lambda_q = lambda_q;
        this.lambda_e = lambda_e;
        this.lambda_bias = lambda_bias;
        this.penalty_weight=penalty_weight;
        this.markovian_steps=markovian_steps;
        this.trade_off = trade_off;
        this.lr = lr;
        this.tol=tol;
        this.max_iter=max_iter;
        this.log_file = log_file;
    }

    //stu,ques,obs,att,res
    public ArrayList<List<Double>> data_set(JSONArray data)
    {
        boolean data_resource = false;
            //student,question,obs,attempt
            if (!data_resource)
            {
                ArrayList<List<Double>> tmp= new ArrayList<List<Double>>();
                for (int i = 0; i < data.size(); i++)
                {
                    JSONArray temp= (JSONArray) data.get(i);
                    ArrayList<Double> each_one=new ArrayList<Double>();
                    for (int j = 0; j < 4; j++)
                    {
                        if(j!=2)
                        {
                            Long tmp_num = (Long) temp.get(j);
                            double a= tmp_num.doubleValue();
                            each_one.add(a);
                        }
                        else
                            {
                                each_one.add((double) temp.get(j));
                            }
                    }
                    double swh = each_one.get(3);
                    each_one.set(3,each_one.get(2));
                    each_one.set(2,each_one.get(1));
                    each_one.set(1,swh);
                    tmp.add(each_one);

                }
                return tmp;
            }
            else {
                ArrayList<List<Double>> tmp= new ArrayList<List<Double>>();
                for (int i = 0; i < data.size(); i++)
                {
                    JSONArray temp = (JSONArray) data.get(i);
                    ArrayList<Double> each_one=new ArrayList<Double>();
                    for (int j = 0; j < 5; j++)
                    {
                        if(j!=2)
                        {
                            Long tmp_num = (Long) temp.get(j);
                            double a= tmp_num.doubleValue();
                            each_one.add(a);
                        }
                        else
                        {
                            each_one.add((double) temp.get(j));
                        }
                    }
                    double swh = each_one.get(3);
                    each_one.set(3,each_one.get(2));
                    each_one.set(2,each_one.get(1));
                    each_one.set(1,swh);
                    tmp.add(each_one);
                }
                return tmp;
            }

    }

}