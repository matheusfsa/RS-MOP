package mop.core;

import java.util.List;

import org.uma.jmetal.problem.Problem;

import mop.problem.BatchProblem;

public interface OffSpringEvaluator<S>{
	List<S> evaluate(List<S> matingPopulation, List<S> solutionList, BatchProblem<S> problem) ;
}
