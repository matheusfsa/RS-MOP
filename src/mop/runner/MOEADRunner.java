package mop.runner;
import mop.algorithm.MOEADBuilder;
import org.json.simple.JSONObject;
import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.algorithm.multiobjective.moead.AbstractMOEAD;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.crossover.IntegerSBXCrossover;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.IntegerPolynomialMutation;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.DoubleProblem;
import org.uma.jmetal.problem.IntegerProblem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.IntegerSolution;
import org.uma.jmetal.util.AbstractAlgorithmRunner;
import org.uma.jmetal.util.AlgorithmRunner;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.ProblemUtils;

import mop.problem.CFProblem;
import requests.Request;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

/**
 * Class for configuring and running the MOEA/DD algorithm
 *
 * @author
 */
public class MOEADRunner extends AbstractAlgorithmRunner {

  public static void main(String[] args) {
	IntegerProblem problem;
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
	    problem = new CFProblem(users.get(i).intValue());
	    
	        /**/
	    double crossoverProbability = 0.9 ;
	    double crossoverDistributionIndex = 20.0 ;
	    crossover = new IntegerSBXCrossover(crossoverProbability, crossoverDistributionIndex) ;

	    double mutationProbability = 1.0 / problem.getNumberOfVariables() ;
	    double mutationDistributionIndex = 20.0 ;
	    mutation = new IntegerPolynomialMutation(mutationProbability, mutationDistributionIndex) ;

	    selection = new BinaryTournamentSelection<IntegerSolution>() ;
	    
	
	    MOEADBuilder builder =  new MOEADBuilder(problem, MOEADBuilder.Variant.MOEADD);
	    builder.setCrossover(crossover)
	            .setMutation(mutation)
	            .setMaxEvaluations(150000)
	            .setPopulationSize(300)
	            .setResultPopulationSize(300)
	            .setNeighborhoodSelectionProbability(0.9)
	            .setMaximumNumberOfReplacedSolutions(1)
	            .setNeighborSize(20)
	            .setFunctionType(AbstractMOEAD.FunctionType.PBI)
	            .setDataDirectory("MOEAD_Weights");
	    algorithm = builder.build();
	
	    AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm)
	            .execute();
	    
	    List<IntegerSolution> population = algorithm.getResult();
	    long computingTime = algorithmRunner.getComputingTime();
    }
}
}
