package mop.problem;

import java.util.List;

import org.uma.jmetal.problem.DoubleProblem;
import org.uma.jmetal.problem.Problem;

public interface BatchProblem<S> extends Problem<S>{
	public List<S> evaluate_batch(List<S> solutionList);
	public List<S> evaluate_batch_offspring(List<S> matingPopulation, List<S> solutionList);
}
