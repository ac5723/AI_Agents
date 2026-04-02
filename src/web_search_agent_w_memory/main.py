from my_ai_learning.crew import MyAiLearning

def run():
    inputs = {
        "topic": "Machine Learning"
    }
    MyAiLearning().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()