import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging format constants
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Action keywords for intent detection
# Maps user-friendly terms to actual ActionType values
ACTION_KEYWORDS = {
    'create': ['create', 'generate', 'make', 'build', 'draft', 'compose', 'write'],
    'summarize': ['summarize', 'summary', 'brief', 'overview', 'recap'],
    'analyze': ['analyze', 'analysis', 'evaluate', 'assess', 'examine'],
    'compare': ['compare', 'comparison', 'versus', 'vs', 'difference'],
    'extract': ['extract', 'retrieve', 'pull'],
    'export': ['export', 'save', 'download', 'output'],
    'search': ['search', 'look for', 'locate'],
    'schedule_meeting': ['schedule', 'book', 'arrange', 'set up', 'organize', 'plan', 'meeting', 'call', 'conference'],
    'file_ticket': ['file ticket', 'create ticket', 'raise ticket', 'log ticket', 'submit ticket', 'new ticket'],
    'request_software': ['request software', 'install software', 'need software', 'software request', 'application request'],
    'escalate_issue': ['escalate', 'urgent', 'critical', 'high priority', 'emergency', 'escalation'],
    'apply_leave': ['apply leave', 'request leave', 'time off', 'vacation', 'pto', 'holiday'],
    'update_documentation': ['update document', 'edit document', 'modify document', 'change document', 'documentation update'],
}


@dataclass
class RetrieverConfig:
    """Configuration for the RAG Retriever."""
    
    vector_db_path: str = field(default_factory=lambda: os.getenv("VECTOR_DB_PATH", "./data/vectordb"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.3")))
    confidence_base: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_BASE", "0.7")))
    confidence_multiplier: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_MULTIPLIER", "0.3")))
    confidence_max: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_MAX", "0.95")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))


@dataclass
class AgentConfig:
    """Configuration class for the Agentic RAG System."""
    
    # LLM Configuration
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"))
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048")))
    
    # Vector Database Configuration
    vector_db_path: str = field(default_factory=lambda: os.getenv("VECTOR_DB_PATH", "./data/vectordb"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    
    # Retrieval Configuration
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.3")))
    confidence_base: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_BASE", "0.7")))
    confidence_multiplier: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_MULTIPLIER", "0.3")))
    confidence_max: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_MAX", "0.95")))
    
    # Document Configuration
    pdf_path: str = field(default_factory=lambda: os.getenv("PDF_PATH", "./data/HCLTech_Annual_Report.pdf"))
    
    # Action Configuration
    enable_actions: bool = field(default_factory=lambda: os.getenv("ENABLE_ACTIONS", "true").lower() == "true")
    action_log_path: str = field(default_factory=lambda: os.getenv("ACTION_LOG_PATH", "./data/actions.json"))
    
    # System Configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._setup_logging()
        self._validate_config()
        self._create_directories()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.llm_api_key:
            self.logger.warning("LLM_API_KEY not found in environment variables")
        
        if self.llm_provider not in ["groq", "anthropic", "openai"]:
            self.logger.warning(f"Unknown LLM provider: {self.llm_provider}. Supported: groq, anthropic, openai")
        
        if self.temperature < 0 or self.temperature > 2:
            self.logger.warning(f"Temperature {self.temperature} is outside recommended range [0, 2]")
        
        if self.max_tokens < 100:
            self.logger.warning(f"max_tokens {self.max_tokens} seems too low")
        
        if self.chunk_size < 100:
            self.logger.warning(f"chunk_size {self.chunk_size} seems too small")
        
        self.logger.info(f"Configuration loaded: Provider={self.llm_provider}, Model={self.model_name}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            os.path.dirname(self.vector_db_path) if self.vector_db_path else None,
            os.path.dirname(self.action_log_path) if self.action_log_path else None,
            os.path.dirname(self.pdf_path) if self.pdf_path else None,
        ]
        
        for directory in directories:
            if directory and directory != "." and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    self.logger.info(f"Created directory: {directory}")
                except Exception as e:
                    self.logger.error(f"Failed to create directory {directory}: {str(e)}")
    
    def get_provider_info(self) -> dict:
        """Get information about the current LLM provider."""
        provider_info = {
            "groq": {
                "name": "Groq",
                "description": "Ultra-fast LLM inference",
                "speed": "500+ tokens/second",
                "models": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "description": "Advanced AI assistant",
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            },
            "openai": {
                "name": "OpenAI",
                "description": "GPT models",
                "models": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"]
            }
        }
        return provider_info.get(self.llm_provider, {"name": "Unknown", "description": "Unknown provider"})
    
    def display_config(self):
        """Display current configuration."""
        print("=" * 70)
        print("  AGENTIC RAG SYSTEM CONFIGURATION")
        print("=" * 70)
        print(f"\n[LLM Configuration]")
        print(f"  Provider:     {self.llm_provider}")
        print(f"  Model:        {self.model_name}")
        print(f"  Temperature:  {self.temperature}")
        print(f"  Max Tokens:   {self.max_tokens}")
        print(f"  API Key:      {'*' * 20}{self.llm_api_key[-10:] if self.llm_api_key else 'NOT SET'}")
        
        print(f"\n[Vector Database]")
        print(f"  Path:         {self.vector_db_path}")
        print(f"  Embedding:    {self.embedding_model}")
        print(f"  Chunk Size:   {self.chunk_size}")
        print(f"  Chunk Overlap: {self.chunk_overlap}")
        
        print(f"\n[Retrieval]")
        print(f"  Top K:        {self.top_k}")
        print(f"  Threshold:    {self.similarity_threshold}")
        
        print(f"\n[Documents]")
        print(f"  PDF Path:     {self.pdf_path}")
        
        print(f"\n[Actions]")
        print(f"  Enabled:      {self.enable_actions}")
        print(f"  Log Path:     {self.action_log_path}")
        
        print(f"\n[System]")
        print(f"  Log Level:    {self.log_level}")
        print("=" * 70)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "vector_db_path": self.vector_db_path,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "confidence_base": self.confidence_base,
            "pdf_path": self.pdf_path,
            "enable_actions": self.enable_actions,
            "action_log_path": self.action_log_path,
            "log_level": self.log_level,
        }
    
    @classmethod
    def from_env(cls):
        """Create AgentConfig from environment variables (alias for default constructor)."""
        return cls()


# For testing the config
if __name__ == "__main__":
    config = AgentConfig()
    config.display_config()
    
    # Test provider info
    print("\nProvider Information:")
    provider_info = config.get_provider_info()
    for key, value in provider_info.items():
        print(f"  {key}: {value}")