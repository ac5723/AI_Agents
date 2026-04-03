from crewai.flow.flow import Flow, start, listen
from crews.research_crew.research_crew import ResearchCrew
from crews.writing_crew.writing_crew import WritingCrew
from crews.review_crew.review_crew import ReviewCrew

class ContentCreationFlow(Flow):

    @start()
    def run_research(self):
        print("=== Starting Research Crew ===")
        result = ResearchCrew().crew().kickoff(
            inputs={"topic": self.state.get("topic", "AI")}
        )
        return result.raw

    @listen(run_research)        # 👈 triggers after research finishes
    def run_writing(self, research_output):
        print("=== Starting Writing Crew ===")
        result = WritingCrew().crew().kickoff(
            inputs={
                "topic": self.state.get("topic", "AI"),
                "research": research_output
            }
        )
        return result.raw

    @listen(run_writing)         # 👈 triggers after writing finishes
    def run_review(self, written_output):
        print("=== Starting Review Crew ===")
        result = ReviewCrew().crew().kickoff(
            inputs={
                "topic": self.state.get("topic", "AI"),
                "article": written_output
            }
        )
        return result.raw