package mop.algorithm;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.util.AlgorithmBuilder;

import mop.core.BatchSolutionListEvaluator;
import mop.core.OffSpringBatchEvaluator;
import mop.core.OffSpringEvaluator;
import mop.problem.BatchProblem;

import java.util.List;


/** Builder class */
public class NSGAIIIBuilder<S extends Solution<?>> implements AlgorithmBuilder<NSGAIII<S>>{
  // no access modifier means access from classes within the same package
  private BatchProblem<S> problem ;
  private int maxIterations ;
  private int populationSize ;
  private CrossoverOperator<S> crossoverOperator ;
  private MutationOperator<S> mutationOperator ;
  private SelectionOperator<List<S>, S> selectionOperator ;

  private BatchSolutionListEvaluator<S> evaluator ;
  private OffSpringEvaluator<S> offSpringEvaluator ;
  
  public OffSpringEvaluator<S> getOffSpringEvaluator() {
	return offSpringEvaluator;
}

/** Builder constructor */
  public NSGAIIIBuilder(BatchProblem<S> problem) {
    this.problem = problem ;
    maxIterations = 250 ;
    populationSize = 100 ;
    evaluator = new BatchSolutionListEvaluator<S>() ;
    offSpringEvaluator = new OffSpringBatchEvaluator<S>();
  }

  public NSGAIIIBuilder<S> setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations ;

    return this ;
  }

  public NSGAIIIBuilder<S> setPopulationSize(int populationSize) {
    this.populationSize = populationSize ;

    return this ;
  }

  public NSGAIIIBuilder<S> setCrossoverOperator(CrossoverOperator<S> crossoverOperator) {
    this.crossoverOperator = crossoverOperator ;

    return this ;
  }

  public NSGAIIIBuilder<S> setMutationOperator(MutationOperator<S> mutationOperator) {
    this.mutationOperator = mutationOperator ;

    return this ;
  }

  public NSGAIIIBuilder<S> setSelectionOperator(SelectionOperator<List<S>, S> selectionOperator) {
    this.selectionOperator = selectionOperator ;

    return this ;
  }

  public NSGAIIIBuilder<S> setSolutionListEvaluator(BatchSolutionListEvaluator<S> evaluator) {
    this.evaluator = evaluator ;

    return this ;
  }

  public BatchSolutionListEvaluator<S> getEvaluator() {
    return evaluator;
  }

  public BatchProblem<S> getProblem() {
    return problem;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public int getPopulationSize() {
    return populationSize;
  }

  public CrossoverOperator<S> getCrossoverOperator() {
    return crossoverOperator;
  }

  public MutationOperator<S> getMutationOperator() {
    return mutationOperator;
  }

  public SelectionOperator<List<S>, S> getSelectionOperator() {
    return selectionOperator;
  }

  public NSGAIII<S> build() {
    return new NSGAIII<S>(this) ;
  }
}