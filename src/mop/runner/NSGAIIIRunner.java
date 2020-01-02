package mop.runner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.json.simple.JSONObject;
import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.AbstractAlgorithmRunner;
import org.uma.jmetal.util.AlgorithmRunner;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.ProblemUtils;
import org.uma.jmetal.util.fileoutput.SolutionListOutput;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;


import mop.algorithm.NSGAIII;
import mop.algorithm.NSGAIIIBuilder;
import mop.problem.RecommenderProblem;
import requests.Request;

public class NSGAIIIRunner extends AbstractAlgorithmRunner {
	  /**
	   * @param args Command line arguments.
	   * @throws java.io.IOException
	   * @throws SecurityException
	   * @throws ClassNotFoundException
	   * Usage: three options
	   *        - org.uma.jmetal.runner.multiobjective.NSGAIIIRunner
	   *        - org.uma.jmetal.runner.multiobjective.NSGAIIIRunner problemName
	   *        - org.uma.jmetal.runner.multiobjective.NSGAIIIRunner problemName paretoFrontFile
	   */
	  @SuppressWarnings("unchecked")
	public static void main(String[] args) throws JMetalException {
		    RecommenderProblem problem;
		    NSGAIII<DoubleSolution> algorithm;
		    CrossoverOperator<DoubleSolution> crossover;
		    MutationOperator<DoubleSolution> mutation;
		    SelectionOperator<List<DoubleSolution>, DoubleSolution> selection;
		    double crossoverProbability = 0.9 ;
		    double crossoverDistributionIndex = 30.0 ;
		    crossover = new SBXCrossover(crossoverProbability, crossoverDistributionIndex) ;

		    
		    
		    
		    selection = new BinaryTournamentSelection<DoubleSolution>();
		    JSONObject data = new JSONObject();
		    JSONObject res = Request.execute("http://127.0.0.1:5000/users", data);    
		    ArrayList<Long> users = (ArrayList<Long>)res.get("response");
		    int n = users.size();
		    for (int i = 0; i < n; i++) {
		    	try {
		    		Runtime.getRuntime().exec("clear");
		    	} catch (IOException e) {
		    		continue;
				}
		    	System.out.println("Faltam " + (i - n) + " usuários");
		    	Long user = users.get(i);
		    	try {
			    	problem = new RecommenderProblem(user);
			    	double mutationProbability = 1.0 / problem.getNumberOfVariables() ;
				    double mutationDistributionIndex = 20.0 ;
				    mutation = new PolynomialMutation(mutationProbability, mutationDistributionIndex) ;
				    
				    algorithm = new NSGAIIIBuilder<>(problem)
				            .setCrossoverOperator(crossover)
				            .setMutationOperator(mutation)
				            .setSelectionOperator(selection)
				            .setMaxIterations(200)
				            .setPopulationSize(20)
				            .build() ;
			
				    AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm)
				            .execute() ;
				    List<DoubleSolution> population = algorithm.getResult() ;
				    ArrayList<ArrayList<Double>> solutions = problem.convert_pop(population);
				    System.out.println(solutions.size());
					JSONObject json = new JSONObject();
					json.put("solucoes", solutions);
					Request.execute("http://127.0.0.1:5000/filtering", json);
				    long computingTime = algorithmRunner.getComputingTime() ;
		    	} catch (Exception e) {
					 return;
				}
			}
		    
		  }
	}
