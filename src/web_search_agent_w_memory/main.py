from my_ai_learning.crew import MyAiLearning

def run():
    inputs = {
        "topic": "Machine Learning"
    }
    result = MyAiLearning().crew().kickoff(inputs=inputs)
    print("\n\n=== CREW RESULT ===")
    print(result)

if __name__ == "__main__":
    run()