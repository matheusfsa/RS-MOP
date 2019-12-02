package mop.problem;

import org.uma.jmetal.solution.DoubleSolution;

public interface BatchDoubleProblem extends BatchProblem<DoubleSolution>{
	 Double getLowerBound(int index) ;
	 Double getUpperBound(int index) ;
}
	