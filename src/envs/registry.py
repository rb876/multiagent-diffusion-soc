
def get_optimality_criterion(name, classifier, **kwargs):
    if name == "classifier_ce":
        from src.envs.optimality import ClassifierCEOptimalityCriterion
        return ClassifierCEOptimalityCriterion(classifier=classifier, **kwargs)
    else:
        raise ValueError(f"Optimality criterion '{name}' not recognized.")