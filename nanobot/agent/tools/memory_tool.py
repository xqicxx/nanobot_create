"""Memory tool for MemU integration."""

from typing import Any

from nanobot.agent.tools.base import Tool


class MemoryRetrieveTool(Tool):
    """Tool to retrieve memories from MemU."""
    
    def __init__(self, memory_adapter: Any):
        self.memory_adapter = memory_adapter
    
    @property
    def name(self) -> str:
        return "retrieve_memory"
    
    @property
    def description(self) -> str:
        return (
            "Retrieve relevant memories from the memory system. "
            "Use this to recall information about the user, past conversations, "
            "important events, or any stored memories. "
            "Call this tool proactively when the user asks about previous topics, "
            "their preferences, or anything that might be in memory."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant memories (e.g., user's name, previous topics, key facts)",
                }
            },
            "required": ["query"],
        }
    
    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        """Execute memory retrieval."""
        if not self.memory_adapter or not self.memory_adapter.enable_memory:
            return "Memory system is not enabled."
        
        try:
            # For now, return all memories as context
            # In the future, this could do semantic search
            import asyncio
            
            # Get memory status to check if there are any memories
            status = await self.memory_adapter.memu_status()
            if not status.get("enabled"):
                return "Memory system is currently disabled."
            
            # Try to read memory files directly
            memory_dir = self.memory_adapter.workspace / ".memu" / "memory"
            if not memory_dir.exists():
                return "No memories stored yet. This is a new conversation."
            
            # Find all memory files
            memories = []
            memory_files = {
                "profile": "个人档案",
                "event": "重要事件", 
                "reminder": "提醒事项",
                "interest": "兴趣爱好",
                "study": "学习记录",
                "activity": "活动记录"
            }
            
            # Try different user directories
            for user_dir in memory_dir.rglob("*"):
                if user_dir.is_dir():
                    for category, label in memory_files.items():
                        file_path = user_dir / f"{category}.md"
                        if file_path.exists():
                            try:
                                content = file_path.read_text(encoding="utf-8").strip()
                                if content and len(content) > 10:
                                    lines = [line.strip() for line in content.split("\n") if line.strip()]
                                    if lines:
                                        summary = "; ".join(lines[:5])  # First 5 lines
                                        memories.append(f"[{label}] {summary}")
                            except Exception:
                                pass
            
            if not memories:
                return "No memories found for this query. This might be a new conversation or the memory system hasn't stored any data yet."
            
            return "Found memories:\n" + "\n".join(f"- {m}" for m in memories[:10])
            
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"


class MemorySaveTool(Tool):
    """Tool to save important information to MemU."""
    
    def __init__(self, memory_adapter: Any):
        self.memory_adapter = memory_adapter
    
    @property
    def name(self) -> str:
        return "save_memory"
    
    @property
    def description(self) -> str:
        return (
            "Save important information to the memory system. "
            "Use this to remember key facts about the user, their preferences, "
            "important events, or any information that should be recalled in future conversations. "
            "Call this tool proactively when the user shares something important about themselves."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The important information to save (e.g., 'User's name is John, they like Python programming')",
                },
                "category": {
                    "type": "string",
                    "description": "Category of the memory (profile, event, reminder, interest, study, activity)",
                    "enum": ["profile", "event", "reminder", "interest", "study", "activity"],
                }
            },
            "required": ["content", "category"],
        }
    
    async def execute(self, **kwargs: Any) -> str:
        content = kwargs.get("content", "")
        category = kwargs.get("category", "activity")
        """Execute memory saving."""
        if not self.memory_adapter or not self.memory_adapter.enable_memory:
            return "Memory system is not enabled."
        
        try:
            import asyncio
            from datetime import datetime
            
            # For now, write directly to file
            # In the future, this should use MemoryAgent.run()
            memory_dir = self.memory_adapter.workspace / ".memu" / "memory"
            user_dir = memory_dir / "nanobot" / "default:unknown:system"
            user_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = user_dir / f"{category}.md"
            
            # Append to existing file or create new
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"\n## {timestamp}\n{content}\n"
            
            if file_path.exists():
                existing = file_path.read_text(encoding="utf-8")
                file_path.write_text(existing + entry, encoding="utf-8")
            else:
                file_path.write_text(f"# {category.capitalize()} Memory\n{entry}", encoding="utf-8")
            
            return f"Successfully saved to {category} memory."
            
        except Exception as e:
            return f"Error saving memory: {str(e)}"
