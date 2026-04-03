from content_creation_multi_crew_pipeline.pipeline import ContentCreationFlow

def run():
    flow = ContentCreationFlow()
    result = flow.kickoff(inputs={"topic": "Future of Artificial Intelligence"})
    print("\n=== FLOW COMPLETE ===")
    print(result)

if __name__ == "__main__":
    run()