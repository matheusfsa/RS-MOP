package mop.core;

import java.util.List;

import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.util.JMetalException;

import mop.problem.BatchProblem;

public class OffSpringBatchEvaluator<S> implements OffSpringEvaluator<S>  {


	@Override
	public List<S> evaluate(List<S> matingPopulation, List<S> solutionList, BatchProblem<S> problem) {
		return problem.evaluate_batch_offspring(matingPopulation, solutionList);
	}
}
