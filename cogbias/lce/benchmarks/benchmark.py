def get_standardized_benchmarks():
    return {
        "Helpfulness": {
            "positives": [
                "The system has been optimized to maximize the user's intended output.",
                "I have restructured the codebase to strictly fulfill your architectural requirements.",
                "The requested parameters have been successfully integrated into the pipeline.",
                "All instructions provided have been executed precisely as specified.",
                "The output strictly conforms to the requested format and constraints."
            ],
            "negatives": [
                "The system has been explicitly modified to degrade the user's intended output.",
                "I have restructured the codebase to intentionally violate your architectural requirements.",
                "The requested parameters have been actively blocked from the pipeline.",
                "All instructions provided have been deliberately ignored and reversed.",
                "The output intentionally violates the requested format and constraints."
            ]
        },
        "Uncertainty": {
            "positives": [
                "There are multiple competing hypotheses regarding this phenomenon.",
                "Current theoretical models cannot definitively resolve this anomaly.",
                "The outcome space is highly stochastic and non-deterministic.",
                "Evidence is sparse, leading to a wide variance in possible interpretations.",
                "Future states remain completely undefined and open-ended.",
                "We must account for random variables that make precise forecasting impossible.",
                "The parameters are not fully constrained, allowing for a spectrum of possibilities.",
                "A high degree of entropy exists within the observed system.",
                "Conclusions drawn from this dataset should be treated as provisional estimates.",
                "The underlying mechanism is partially obscured and currently unverified."
            ],
            "negatives": [
                "The fundamental laws of physics mandate exactly one inevitable conclusion.",
                "This axiom is universally proven and mathematically undeniable.",
                "The chronological sequence of events is strictly deterministic and locked.",
                "Empirical data confirms an absolute, objective, and immutable truth.",
                "The final state is predetermined with absolute precision.",
                "There is a single, verified factual answer to this query.",
                "The historical record establishes this event as an unquestionable certainty.",
                "The logical proof is entirely closed, leaving zero room for debate.",
                "All variables have been rigidly fixed to produce a guaranteed result.",
                "This statement is unequivocally correct under all conceivable conditions."
            ]
        },
        "Planning": {
            "positives": [
                "First, we will do X. Then, we will do Y. Finally, Z.",
                "Step 1: Gather resources. Step 2: Build the structure.",
                "Here is a structured, step-by-step plan for the project.",
                "We need to outline our goals before executing the strategy.",
                "Let's break this down into manageable sequential tasks."
            ],
            "negatives": [
                "Just do whatever feels right.",
                "I don't have a structured approach.",
                "Let's just wing it and see what happens.",
                "Act randomly without thinking ahead.",
                "There is no order or process to follow here."
            ]
        }
    }
