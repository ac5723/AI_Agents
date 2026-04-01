from my_ai_learning.crew import MyAiLearning

def run():
    inputs = {
        "topic": "Artificial Intelligence"  # 👈 change this to anything!
    }
    MyAiLearning().crew().kickoff(inputs=inputs)

# if run == "__main__":
#     run()
run()

