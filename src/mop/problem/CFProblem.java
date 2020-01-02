package mop.problem;

import java.util.ArrayList;
import java.util.List;

import org.json.simple.JSONObject;
import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.algorithm.multiobjective.moead.AbstractMOEAD;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.crossover.IntegerSBXCrossover;
import org.uma.jmetal.operator.impl.mutation.IntegerPolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.IntegerProblem;
import org.uma.jmetal.problem.impl.AbstractIntegerProblem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.IntegerSolution;
import org.uma.jmetal.util.AlgorithmRunner;

import mop.algorithm.MOEADBuilder;
import requests.Request;

public class CFProblem extends AbstractIntegerProblem {
	private List<Long> lowerLimit ;
	private List<Long> upperLimit ;
	private int n_variables;
	public CFProblem(long user) {
		JSONObject json = new JSONObject();
		json.put("user", user);
		Request.execute("http://127.0.0.1:5000/user", json);
		this.setNumberOfVariables();
		this.setLowerBound();
		this.setUpperBound();
		
	}
	@Override
	public int getNumberOfObjectives() {
		// TODO Auto-generated method stub
		return 2;
	}
	
	@Override
	public int getNumberOfVariables() {
		return 15;
	}
	public void setNumberOfVariables() {
		JSONObject json = new JSONObject();
		JSONObject res = Request.execute("http://127.0.0.1:5000/n-variables", json);
		ArrayList<Long> n_variables = (ArrayList<Long>)res.get("response");
		this.n_variables =  n_variables.get(0).intValue();
	}
	
	@Override
	public Integer getUpperBound(int index) {
		return upperLimit.get(index).intValue();
	}

	@Override
	public Integer getLowerBound(int index) {
		return lowerLimit.get(index).intValue();
	}
	
	public void setLowerBound() {
		JSONObject json = new JSONObject();
		JSONObject res = Request.execute("http://127.0.0.1:5000/min", json);
		
		lowerLimit = (ArrayList<Long>)res.get("response");
	}

	
	
	public void setUpperBound() {
		JSONObject json = new JSONObject();
		JSONObject res = Request.execute("http://127.0.0.1:5000/max", json);
		upperLimit = (ArrayList<Long>)res.get("response");
	}
	public ArrayList<Integer> is2i(IntegerSolution ds){
		ArrayList<Integer> d = new ArrayList<Integer>();
		for(int i = 0; i < this.n_variables; i++) {
			d.add(ds.getVariableValue(i)); 
		}
		return d;
	} 
	public ArrayList<ArrayList<Integer>> convert_pop(List<IntegerSolution> pop){
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		for (IntegerSolution ds: pop) {
			res.add(is2i(ds));
			
		}
		return  res;
	}
	@Override
	public void evaluate(IntegerSolution s) {
		ArrayList<Integer> sol = is2i(s);
		JSONObject json = new JSONObject();
		json.put("solution", sol);
		JSONObject res = Request.execute("http://127.0.0.1:5000/evaluate-solution", json);
		ArrayList<Double> obj = (ArrayList<Double>) res.get("response");
		s.setObjective(0, obj.get(0));
		s.setObjective(1, obj.get(1));
		
	}
	
	public static void main(String[] args) {
		CFProblem problem;
	    Algorithm<List<IntegerSolution>> algorithm;
	    String problemName;
	    String referenceParetoFront = "";
	    
	    JSONObject data = new JSONObject();
	    JSONObject res = Request.execute("http://127.0.0.1:5000/users", data);    
	    ArrayList<Long> users = (ArrayList<Long>)res.get("response");
	    int n = users.size();
	    CrossoverOperator<IntegerSolution> crossover;
	    MutationOperator<IntegerSolution> mutation;
	    SelectionOperator<List<IntegerSolution>, IntegerSolution> selection;
	    for (int i = 0; i < n; i++) {
	    	System.out.println("Faltam " + (i - n) + " usuários");
		    problem = new CFProblem(users.get(i).intValue());
		    
		        
		    double crossoverProbability = 0.9 ;
		    double crossoverDistributionIndex = 20.0 ;
		    crossover = new IntegerSBXCrossover(crossoverProbability, crossoverDistributionIndex) ;

		    double mutationProbability = 1.0 / problem.getNumberOfVariables() ;
		    double mutationDistributionIndex = 20.0 ;
		    mutation = new IntegerPolynomialMutation(mutationProbability, mutationDistributionIndex) ;

		    selection = new BinaryTournamentSelection<IntegerSolution>() ;

		    algorithm = new NSGAIIBuilder<IntegerSolution>(problem, crossover, mutation, 20)
		            .setSelectionOperator(selection)
		            .setMaxEvaluations(200)
		            .build() ;

		    AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm)
		            .execute() ;

		    List<IntegerSolution> population = algorithm.getResult() ;
		    ArrayList<ArrayList<Integer>> solutions = problem.convert_pop(population);
		    long computingTime = algorithmRunner.getComputingTime();
		    JSONObject json = new JSONObject();
			json.put("solucoes", solutions);
			Request.execute("http://127.0.0.1:5000/filtering", json);
	    }
		
	}
}
