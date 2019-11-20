import settings
import operator
from primitives import (
    mate,
    mutate,
    create_attribute,
    evaluate_individual,
)
from deap import (
    gp,
    base,
    tools,
    creator,
    algorithms,
)


pset = gp.PrimitiveSet("ind", settings.INPUT_NUMBER)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)

for index in range(2, settings.INPUT_NUMBER+2):
    pset.addPrimitive(max, index)
    pset.addPrimitive(min, index)
    pset.addTerminal(3)


creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register('attribute', 
                 create_attribute, 
                 pset=pset)
toolbox.register('individual',
                 tools.initRepeat,
                 creator.Individual, 
                 toolbox.attribute,
                 settings.OUTPUT_NUMBER)
toolbox.register("population",
                 tools.initRepeat,
                 list, 
                 toolbox.individual)
toolbox.register('compile', 
                 gp.compile, 
                 pset=pset)
toolbox.register('evaluate', 
                 evaluate_individual,
                 toolbox=toolbox)
toolbox.register('select', 
                 tools.selTournament, 
                 tournsize=settings.TOURNSIZE)
toolbox.register("mate",
                 mate,
                 creator=creator)
toolbox.register("mutate",
                 mutate,
                 pset=pset)


population = toolbox.population(n=settings.POPULATION)
hof = tools.HallOfFame(1)

population, log = algorithms.eaSimple(population=population,
                                      toolbox=toolbox,
                                      cxpb=settings.MATING_PROBABILITY,
                                      mutpb=settings.MUTATE_PROBABILITY,
                                      ngen=settings.NUMBER_OF_GENERATIONS,
                                      halloffame=hof,
                                      verbose=settings.VERBOSE)
