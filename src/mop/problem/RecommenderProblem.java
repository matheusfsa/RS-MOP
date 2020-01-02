package mop.problem;

import java.util.ArrayList;
import java.util.List;

import org.json.simple.JSONObject;
import org.uma.jmetal.solution.DoubleSolution;

import requests.Request;
@SuppressWarnings({ "unchecked", "serial" })
public class RecommenderProblem implements BatchDoubleProblem{
	private int n_variables;
	private ArrayList<Double> lowerBound;
	private ArrayList<Double> upperBound;
	
	
	public RecommenderProblem(long user) {
		JSONObject json = new JSONObject();
		json.put("user", user);
		Request.execute("http://127.0.0.1:5000/user", json);
		this.setNumberOfVariables();
		this.setLowerBound();
		this.setUpperBound();
		
	}
	@Override
	public List<DoubleSolution> evaluate_batch(List<DoubleSolution> solutionList) {
		ArrayList<ArrayList<Double>> solutions = convert_pop(solutionList);
		JSONObject json = new JSONObject();
		json.put("solucoes", solutions);
		JSONObject res = Request.execute("http://127.0.0.1:5000/evaluate-solutions", json);
		ArrayList<ArrayList<Double>> objetivos = (ArrayList<ArrayList<Double>>)res.get("response");
		for(int i = 0; i < solutionList.size(); i++) {
			DoubleSolution s = solutionList.get(i);
			for(int j = 0; j < getNumberOfObjectives(); j++)
				s.setObjective(j, objetivos.get(i).get(j));
			solutionList.set(i, s);
		}
		
		return solutionList;
	}

	@Override
	public List<DoubleSolution> evaluate_batch_offspring(List<DoubleSolution> matingPopulation,
			List<DoubleSolution> solutionList) {
		ArrayList<ArrayList<Double>> solutions = convert_pop(solutionList);
		ArrayList<ArrayList<Double>> mating = convert_pop(matingPopulation);
		JSONObject json = new JSONObject();
		json.put("solucoes", solutions);
		json.put("mating", mating);
		JSONObject res = Request.execute("http://127.0.0.1:5000/evaluate-solutions-offspring", json);
		ArrayList<ArrayList<Double>> objetivos = (ArrayList<ArrayList<Double>>)res.get("response");
		for(int i = 0; i < solutionList.size(); i++) {
			DoubleSolution s = solutionList.get(i);
			for(int j = 0; j < getNumberOfObjectives(); j++)
				s.setObjective(j, objetivos.get(i).get(j));
			solutionList.set(i, s);
		}
		
		return solutionList;
	}

	@Override
	public DoubleSolution createSolution() {
		// TODO Auto-generated method stub
		return new DefaultDoubleSolution(this);
	}

	@Override
	public void evaluate(DoubleSolution arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return "Recommender Problem";
	}

	@Override
	public int getNumberOfConstraints() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getNumberOfObjectives() {
		// TODO Auto-generated method stub
		return 3;
	}

	@Override
	public int getNumberOfVariables() {
		return this.n_variables;
	}
	
	public void setNumberOfVariables() {
		JSONObject json = new JSONObject();
		JSONObject res = Request.execute("http://127.0.0.1:5000/n-variables", json);
		ArrayList<Long> n_variables = (ArrayList<Long>)res.get("response");
		this.n_variables =  n_variables.get(0).intValue();
	}
	
	@Override
	public Double getLowerBound(int index) {
		return this.lowerBound.get(index);
	}
	
	public void setLowerBound() {
		JSONObject json = new JSONObject();
		JSONObject res = Request.execute("http://127.0.0.1:5000/min", json);
		this.lowerBound = (ArrayList<Double>)res.get("response");
	}

	@Override
	public Double getUpperBound(int index) {
		return this.upperBound.get(index);
	}
	
	public void setUpperBound() {
		JSONObject json = new JSONObject();
		JSONObject res = Request.execute("http://127.0.0.1:5000/max", json);
		this.upperBound = (ArrayList<Double>)res.get("response");
	}
	
	public ArrayList<Double> ds2d(DoubleSolution ds){
		ArrayList<Double> d = new ArrayList<Double>();
		for(int i = 0; i < this.n_variables; i++) {
			d.add(ds.getVariableValue(i)); 
		}
		return d;
	} 
	public ArrayList<ArrayList<Double>> convert_pop(List<DoubleSolution> pop){
		
		ArrayList<ArrayList<Double>> res = new ArrayList<ArrayList<Double>>();
		for (DoubleSolution arrayList : pop) {
			res.add(ds2d(arrayList));
		}
		return res;
	}

	
	public static void main(String[] args) {
		RecommenderProblem rp = new RecommenderProblem(1);
		List<DoubleSolution> pop = new ArrayList<DoubleSolution>();
		for(int i = 0; i < 10; i++) {
			pop.add(rp.createSolution());
			System.out.println(rp.ds2d(pop.get(i)));
		}
		rp.evaluate_batch(pop);
		for(int i = 0; i < 10; i++) {
			DoubleSolution ds = pop.get(i);
			System.out.println("Precision:"+ds.getObjective(0) + "; Diversity:" + ds.getObjective(1) + "; Novelty:"+ ds.getObjective(2));
		}
		List<DoubleSolution> pop2 = new ArrayList<DoubleSolution>();
		for(int i = 0; i < 10; i++) {
			pop2.add(rp.createSolution());
			System.out.println(rp.ds2d(pop2.get(i)));
		}
		rp.evaluate_batch_offspring(pop, pop2);
		for(int i = 0; i < 10; i++) {
			DoubleSolution ds = pop2.get(i);
			System.out.println("Precision:"+ds.getObjective(0) + "; Diversity:" + ds.getObjective(1) + "; Novelty:"+ ds.getObjective(2));
		}
	}
}
