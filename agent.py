"""
Main Agent Module - Orchestrates the complete Agentic RAG system
"""
import json
import logging
import sys
from typing import Dict, Any, List
from datetime import datetime

from config import AgentConfig, LOG_FORMAT, LOG_DATE_FORMAT
from intent_detector import IntentDetector
from retriever import RAGRetriever
from llm_handler import LLMHandler
from actions import ActionExecutor, extract_action_parameters


class AgenticRAGAssistant:
    """Main Agentic RAG Assistant for Enterprise queries."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self._setup_logging()
        
        # Initialize components
        self.intent_detector = IntentDetector()
        self.retriever = RAGRetriever(config)
        self.llm_handler = LLMHandler(config)
        self.action_executor = ActionExecutor() if config.enable_actions else None
        
        self.logger.info("Agentic RAG Assistant initialized")
    
    def _setup_logging(self):
        """Configure logging with file and console output."""
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT,
            handlers=[
                logging.FileHandler('agent.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize the assistant by loading required resources."""
        try:
            self.logger.info("="*60)
            self.logger.info("Initializing Agentic RAG Assistant...")
            self.logger.info("="*60)
            
            # Load vector store and embeddings
            if self.retriever.load_vectorstore() is None:
                self.logger.error("Failed to load vector store")
                return False
            
            self.logger.info("Assistant initialized successfully")
            self.logger.info("="*60)
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing assistant: {str(e)}")
            return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main method to process user queries."""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info(f"Processing query: {query}")
            self.logger.info("="*60)
            
            # Step 1: Detect intent
            intent = self.intent_detector.detect_intent(query)
            self.logger.info(f"Intent: {intent['intent_type']} (confidence: {intent['confidence']})")
            
            # Step 2: Retrieve context from vector store
            context, pages, retrieval_confidence = self.retriever.retrieve_context(query)
            
            result = {
                "query": query,
                "intent": intent,
                "retrieval": {
                    "pages": pages,
                    "confidence": retrieval_confidence,
                    "context_length": len(context)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 3: Process based on intent
            if intent["intent_type"] == "action" and self.config.enable_actions:
                result.update(self._process_action_query(query, intent, context, pages))
            else:
                result.update(self._process_information_query(query, context, pages))
            
            self.logger.info("Query processed successfully")
            self.logger.info("="*60)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_information_query(self, query: str, context: str, pages: List[int]) -> Dict[str, Any]:
        """Process information retrieval query."""
        self.logger.info("Processing as information query...")
        
        # Generate LLM response
        answer = self.llm_handler.generate_response(query, context, pages)
        
        return {
            "response_type": "information",
            "answer": answer,
            "context_preview": context[:300] + "..." if len(context) > 300 else context
        }
    
    def _process_action_query(self, query: str, intent: Dict[str, Any], context: str, pages: List[int]) -> Dict[str, Any]:
        """Process action execution query."""
        self.logger.info(f"Processing as action query: {intent['action_type']}")
        
        # Extract parameters for the action
        parameters = extract_action_parameters(query, intent["action_type"])
        self.logger.info(f"Extracted parameters: {parameters}")
        
        # Execute the action
        action_result = self.action_executor.execute_action(intent["action_type"], parameters)
        
        # Generate contextual response with action results
        explanation = self.llm_handler.generate_with_actions(query, context, pages, action_result)
        
        return {
            "response_type": "action",
            "action": action_result,
            "explanation": explanation,
            "context_preview": context[:300] + "..." if len(context) > 300 else context
        }
    
    def get_action_history(self) -> List[Dict]:
        """Get history of all executed actions."""
        if self.action_executor:
            return self.action_executor.get_action_history()
        return []
    
    def clear_action_history(self):
        """Clear the action execution history."""
        if self.action_executor:
            self.action_executor.clear_history()
            self.logger.info("Action history cleared")
    
    def export_session_log(self, filepath: str = "session_log.json") -> bool:
        """Export the complete session log including actions."""
        try:
            session_data = {
                "session_timestamp": datetime.now().isoformat(),
                "configuration": {
                    "model": self.config.model_name,
                    "llm_provider": self.config.llm_provider,
                    "vector_db": self.config.vector_db_path,
                    "top_k": self.config.top_k,
                    "actions_enabled": self.config.enable_actions
                },
                "actions": self.get_action_history(),
                "action_count": len(self.get_action_history())
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info(f"Session log exported to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting session log: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the assistant's operations."""
        actions = self.get_action_history()
        
        action_types = {}
        for action in actions:
            action_type = action.get("action_type", "unknown")
            action_types[action_type] = action_types.get(action_type, 0) + 1
        
        return {
            "total_actions": len(actions),
            "actions_by_type": action_types,
            "configuration": {
                "llm_provider": self.config.llm_provider,
                "model": self.config.model_name,
                "vector_db": self.config.vector_db_path,
                "top_k": self.config.top_k
            }
        }
    
    def interactive_mode(self):
        """Run the assistant in interactive CLI mode."""
        print("\n" + "="*60)
        print("HCLTech Agentic Enterprise Assistant - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  'exit' or 'quit' - Exit the assistant")
        print("  'history' - Show action history")
        print("  'stats' - Show statistics")
        print("  'clear' - Clear action history")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit']:
                    print("\nExiting... Goodbye!")
                    break
                
                if query.lower() == 'history':
                    history = self.get_action_history()
                    print(f"\n{'='*60}")
                    print(f"Action History ({len(history)} actions)")
                    print("="*60)
                    for i, action in enumerate(history, 1):
                        print(f"\n{i}. {action['action_type']} - {action['timestamp']}")
                        print(f"   Status: {action['result'].get('status', 'unknown')}")
                    print("="*60)
                    continue
                
                if query.lower() == 'stats':
                    stats = self.get_statistics()
                    print(f"\n{'='*60}")
                    print("Statistics")
                    print("="*60)
                    print(json.dumps(stats, indent=2))
                    print("="*60)
                    continue
                
                if query.lower() == 'clear':
                    self.clear_action_history()
                    print("\nAction history cleared.")
                    continue
                
                # Process the query
                result = self.process_query(query)
                
                # Display response
                print(f"\n{'='*60}")
                print("Assistant Response")
                print("="*60)
                
                if result.get("error"):
                    print(f"Error: {result['error']}")
                else:
                    print(f"\nIntent: {result['intent']['intent_type']}")
                    if result['retrieval']['pages']:
                        print(f"Sources: Pages {result['retrieval']['pages']}")
                    print(f"Confidence: {result['retrieval']['confidence']}")
                    
                    if result.get('response_type') == 'action':
                        print(f"\n{'Action Executed':^60}")
                        print("-"*60)
                        action = result['action']
                        print(f"Type: {action['action_type']}")
                        print(f"Status: {action['status']}")
                        print(f"Action ID: {action['action_id']}")
                        print("\nDetails:")
                        print(json.dumps(action['details'], indent=2))
                        print("\n" + "-"*60)
                        print(f"Explanation:\n{result['explanation']}")
                    else:
                        print(f"\nAnswer:\n{result['answer']}")
                
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\nError: {str(e)}")


def main():
    """Main entry point for the agent."""
    config = AgentConfig.from_env()
    assistant = AgenticRAGAssistant(config)
    
    if not assistant.initialize():
        print("Failed to initialize assistant. Please check logs.")
        sys.exit(1)
    
    assistant.interactive_mode()
    assistant.export_session_log()
    print("\nSession log saved to 'session_log.json'")


if __name__ == "__main__":
    main()