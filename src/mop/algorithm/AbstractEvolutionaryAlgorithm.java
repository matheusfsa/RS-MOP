package mop.algorithm;

import java.util.List;

import org.uma.jmetal.algorithm.Algorithm;

import mop.problem.BatchProblem;

@SuppressWarnings("serial")
public abstract class AbstractEvolutionaryAlgorithm<S, R>  implements Algorithm<R>{
  protected List<S> population;
  protected BatchProblem<S> problem ;

  public List<S> getPopulation() {
    return population;
  }
  public void setPopulation(List<S> population) {
    this.population = population;
  }

  public void setProblem(BatchProblem<S> problem) {
    this.problem = problem ;
  }
  public BatchProblem<S> getProblem() {
    return problem ;
  }

  protected abstract void initProgress();

  protected abstract void updateProgress();

  protected abstract boolean isStoppingConditionReached();

  protected abstract  List<S> createInitialPopulation() ;

  protected abstract List<S> evaluatePopulation(List<S> population);
  
  protected abstract List<S> evaluateOffSpringPopulation(List<S> population, List<S> offspringPopulation);

  protected abstract List<S> selection(List<S> population);

  protected abstract List<S> reproduction(List<S> population);

  protected abstract List<S> replacement(List<S> population, List<S> offspringPopulation);

  @Override public abstract R getResult();

  @Override public void run() {
    List<S> offspringPopulation;
    List<S> matingPopulation;

    population = createInitialPopulation();
    population = evaluatePopulation(population);
    initProgress();
    while (!isStoppingConditionReached()) {
      matingPopulation = selection(population);
      offspringPopulation = reproduction(matingPopulation);
      matingPopulation = evaluateOffSpringPopulation(offspringPopulation, matingPopulation);
      offspringPopulation = evaluateOffSpringPopulation(matingPopulation, offspringPopulation);
      population = replacement(population, offspringPopulation);
      updateProgress();
    }
  }
}