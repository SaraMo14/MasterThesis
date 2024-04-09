from pgeon import policy_graph

class PolicyGraphEvaluator:
    def __init__(self, policy_graph: policy_graph):
        self.policy_graph = policy_graph

    def evaluate_static_metrics(self):
        # Implement static evaluation metrics here
        pass

    def evaluate_intention_metrics(self, desires):
        # Implement intention-based evaluation metrics here, utilizing `desires`
        pass

# Example usage:
#pg = PolicyGraph(...)
#evaluator = PolicyGraphEvaluator(pg)
#evaluator.evaluate_static_metrics()
#evaluator.evaluate_intention_metrics(desires)
