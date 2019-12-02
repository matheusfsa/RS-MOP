package mop.core;

import java.util.List;

import mop.problem.BatchProblem;

public class BatchSolutionListEvaluator<S>{
	public List<S> evaluate(List<S> solutionList, BatchProblem<S> problem) {
		return problem.evaluate_batch(solutionList);
	}
}
