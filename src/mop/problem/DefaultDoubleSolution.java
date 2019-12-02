package mop.problem;

import org.uma.jmetal.problem.DoubleProblem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.impl.AbstractGenericSolution;

import java.util.HashMap;
import java.util.Map;

/**
 * Defines an implementation of a double solution
 *
 * @author Antonio J. Nebro <antonio@lcc.uma.es>
 */
@SuppressWarnings("serial")
public class DefaultDoubleSolution 
    extends AbstractGenericSolution<Double, BatchDoubleProblem>
    implements DoubleSolution {

  /** Constructor */
 
  public DefaultDoubleSolution(BatchDoubleProblem problem) {
	    super(problem) ;

	    initializeDoubleVariables();
	    initializeObjectiveValues();
	  }

  /** Copy constructor */
  public DefaultDoubleSolution(DefaultDoubleSolution solution) {
    super(solution.problem) ;

    for (int i = 0; i < problem.getNumberOfVariables(); i++) {
      setVariableValue(i, solution.getVariableValue(i));
    }

    for (int i = 0; i < problem.getNumberOfObjectives(); i++) {
      setObjective(i, solution.getObjective(i)) ;
    }

    attributes = new HashMap<Object, Object>(solution.attributes) ;
  }

  @Override
  public Double getUpperBound(int index) {
    return problem.getUpperBound(index);
  }

  @Override
  public Double getLowerBound(int index) {
    return problem.getLowerBound(index) ;
  }

  @Override
  public DefaultDoubleSolution copy() {
    return new DefaultDoubleSolution(this);
  }

  @Override
  public String getVariableValueString(int index) {
    return getVariableValue(index).toString() ;
  }
  
  private void initializeDoubleVariables() {
    for (int i = 0 ; i < problem.getNumberOfVariables(); i++) {
      Double value = randomGenerator.nextDouble(getLowerBound(i), getUpperBound(i)) ;
      setVariableValue(i, value) ;
    }
  }

@Override
public Map<Object, Object> getAttributes() {
	// TODO Auto-generated method stub
	return null;
}
}