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
        """Execute memory retrieval using memU semantic search."""
        if not self.memory_adapter or not self.memory_adapter.enable_memory:
            return "Memory system is not enabled."

        try:
            import asyncio

            # Get memory status
            status = await self.memory_adapter.memu_status()
            if not status.get("enabled"):
                return "Memory system is currently disabled."

            # Check if MemoryAgent is available for semantic retrieval
            if not self.memory_adapter._memory_agent:
                return "MemoryAgent not initialized. No memories to retrieve."

            # Use memU's built-in semantic retrieval
            try:
                result = await asyncio.wait_for(
                    self.memory_adapter._memory_agent.retrieve(
                        query=query,
                        method="rag",  # Fast RAG mode
                        user={"user_id": "default"},
                    ),
                    timeout=10.0,
                )

                items = result.get("items", [])
                categories = result.get("categories", [])

                if not items and not categories:
                    return f"No memories found for query: '{query}'"

                # Format results
                lines = []
                if categories:
                    for cat in categories[:5]:
                        name = cat.get("name", "")
                        summary = cat.get("summary", "")[:200]
                        if summary:
                            lines.append(f"[分类: {name}] {summary}")

                if items:
                    for item in items[:5]:
                        content = item.get("content", "")[:200]
                        memory_type = item.get("memory_type", "")
                        if content:
                            lines.append(f"[{memory_type}] {content}")

                if not lines:
                    return f"No memories found for query: '{query}'"

                return "找到记忆:\n" + "\n".join(lines[:5])  # Top 5 results

            except asyncio.TimeoutError:
                return "Memory retrieval timed out. Please try again."
            except Exception as e:
                return f"Error retrieving memories: {str(e)}"

        except Exception as e:
            return f"Error: {str(e)}


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
